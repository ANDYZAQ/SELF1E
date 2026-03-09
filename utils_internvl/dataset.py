import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.internvl3 import conversation as conversation_lib

from .img_loading import load_image

from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .grefer import G_REFER
from .refer_seg_dataset import ReferSegDataset, ReferSegDatasetSeq
from .sem_seg_dataset import SemSegDataset, SemSegDatasetSeq
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN, 
                    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN, 
                    IGNORE_INDEX)
from .vqa_dataset import VQADataset, AddVQADataset


def get_cus_attn_mask_seg(input_embed, text_mask, selected, seg_pos):
    # Let seg token visible to img token while bidirectional attention on image token
    B, N = input_embed.shape[:2]
    device = input_embed.device
    
    # default causal mask and text mask
    causal_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
    text_mask_expanded = text_mask[:, None, None, :].expand(-1, 1, N, -1)
    attn_mask = text_mask_expanded | causal_mask[None, None, :, :].expand(B, 1, -1, -1)
    
    # image token bidirectional attention: remove causal constraint
    img_to_img_mask = (selected[:, None, :, None] & selected[:, None, None, :]).expand(-1, 1, -1, -1) # [B, 1, N, N]
    # confine img-to-img bidirectional attention only when there is seg token
    is_img_for_seg = seg_pos.any(dim=1) # [B]
    img_to_img_mask = img_to_img_mask & is_img_for_seg[:, None, None, None].expand(-1, 1, N, N)
    attn_mask = torch.where(img_to_img_mask, text_mask_expanded, attn_mask)

    # latter tokens cannot see seg token
    # attn_mask = attn_mask & ~seg_pos[:, None, None, :].expand(-1, 1, -1, -1)

    # image token see seg token bidirectional attention
    img_to_seg_mask = (selected[:, None, :, None] & seg_pos[:, None, None, :]).expand(-1, 1, -1, -1)
    attn_mask = torch.where(img_to_seg_mask, text_mask_expanded, attn_mask)
    
    # ensure self-attention
    diag_mask = torch.eye(N, device=device).bool()
    attn_mask[:, :, diag_mask] = False

    # seg token cannot see image token
    # mask = (seg_pos[:, None, :, None] & selected[:, None, None, :]).expand(-1, 1, -1, -1)  # [B, 1, N, N]
    # attn_mask = attn_mask & mask
    
    # convert to -inf format
    return attn_mask.float().masked_fill(attn_mask.bool(), float('-inf'))

def get_causal_attn_mask(input_embed, text_mask, selected):
    B, N = input_embed.shape[:2]
    device = input_embed.device
    
    causal_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
    text_mask_expanded = text_mask[:, None, None, :].expand(-1, 1, N, -1)
    attn_mask = text_mask_expanded | causal_mask[None, None, :, :].expand(B, 1, -1, -1)
    return attn_mask.float().masked_fill(attn_mask.bool(), float('-inf'))

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, num_image_token=1, local_rank=-1
):
    image_path_list = []
    images_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    target_aspect_ratios_list = []
    for (
        image_path,
        images,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        target_aspect_ratios,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.extend([mask.unsqueeze(0) for mask in masks.float()] if len(masks) else [masks.float()])
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        target_aspect_ratios_list.append(target_aspect_ratios)
        inferences.append(inference)
        if len(masks):
            assert len(masks) == len(conversations), f"masks{len(masks)}, conversations{len(conversations)}"
        else:
            assert 1 == len(conversations), f"conversations{len(conversations)}"

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)

    assert len(masks_list)==len(conversation_list), f"masks{len(masks_list)}, conversations{len(conversation_list)}"

    input_ids_list, labels_list, attention_mask_list, position_ids_list, image_flags_list = [], [], [], [], []
    offset_idx = 0
    recon_image_list = []
    recon_resize_list = []
    recon_target_aspect_ratios_list = []
    for idx, conversation in enumerate(conversation_list):
        if offset_list[offset_idx] == idx:
            image = images_list[offset_idx]
            offset_idx += 1
        num_patches_list = [image.shape[0]] if image is not None else []

        query = conversation
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        if add_bos_token:  # for InternLM series
            conversation_list[0] = tokenizer.bos_token + conversation_list[0]
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        if add_bos_token:  # for InternLM series
            input_ids = input_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]

        labels = input_ids.clone()
        start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        start_ids = (input_ids[0] == start_token_id).nonzero()
        # pad on system prompt
        labels[0][:start_ids[1]] = IGNORE_INDEX
        for seq in range(1, len(start_ids)):
            if seq % 2 == 1:
                # ignore human response
                labels[0][start_ids[seq]:start_ids[seq+1]] = IGNORE_INDEX
            else:
                # ignore the start of the assistant response
                labels[0][start_ids[seq]:start_ids[seq]+3] = IGNORE_INDEX
                # ignore the \n of the assistant response
                if seq != len(start_ids) - 1:
                    labels[0][start_ids[seq+1]-1:start_ids[seq+1]] = IGNORE_INDEX
                else:
                    labels[0][-1:] = IGNORE_INDEX

        input_ids = input_ids[0][:tokenizer.model_max_length]
        labels = labels[0][:tokenizer.model_max_length]
        attention_mask = attention_mask[0][:tokenizer.model_max_length]
        position_ids = attention_mask.unsqueeze(0).long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.unsqueeze(0) == 0, 1)
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids[0])
        image_flags_list.append(torch.tensor([1] * image.shape[0], dtype=torch.long))

        recon_image_list.append(image)
        recon_resize_list.append(resize_list[offset_idx-1])
        recon_target_aspect_ratios_list.append(target_aspect_ratios_list[offset_idx-1])

    # padding
    max_length = max([input_ids.shape[0] for input_ids in input_ids_list])
    for i in range(len(input_ids_list)):
        padding_length = max_length - input_ids_list[i].shape[0]
        input_ids_list[i] = torch.cat([input_ids_list[i], torch.zeros(padding_length, dtype=torch.long)])
        labels_list[i] = torch.cat([labels_list[i], torch.ones(padding_length, dtype=torch.long) * IGNORE_INDEX])
        attention_mask_list[i] = input_ids_list[i].ne(0)
        position_ids_list[i] = torch.cat([position_ids_list[i], torch.zeros(padding_length, dtype=torch.long)])
    
    pixel_values = torch.cat(recon_image_list, dim=0)

    # generate custom attention mask
    input_ids_tensor = torch.stack(input_ids_list, dim=0)
    attention_mask_tensor = torch.stack(attention_mask_list, dim=0)
    custom_causal_mask = None
    
    # check if there is image token and generate custom mask
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    selected = (input_ids_tensor == img_context_token_id)
    if selected.any():
        B, N = input_ids_tensor.shape
        fake_embed = torch.zeros(B, N, 1)
        text_mask = ~attention_mask_tensor
        seg_pos = (input_ids_tensor == tokenizer.convert_tokens_to_ids('[SEG]'))
        if seg_pos.any():
            custom_causal_mask = get_cus_attn_mask_seg(fake_embed, text_mask, selected, seg_pos)
        else:
            custom_causal_mask = get_causal_attn_mask(fake_embed, text_mask, selected)
    
    # choose to use custom mask or default mask
    final_attention_mask = custom_causal_mask if custom_causal_mask is not None else attention_mask_tensor

    input_dict = {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
        "attention_mask": final_attention_mask, 
        "position_ids": torch.stack(position_ids_list, dim=0),
        "image_flags": torch.cat(image_flags_list, dim=0),
        "pixel_values": pixel_values,
    }

    input_dict.update({
        "image_paths": image_path_list,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": recon_resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "target_aspect_ratios_list": recon_target_aspect_ratios_list,
        "conversation_list": conversation_list,
        "input_ids_decoded": [[tokenizer.decode(token_id.item()) for token_id in input_ids if token_id != 0] for input_ids in input_ids_list],
    })

    return input_dict


ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train_balanced.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/ai2d_train_12k.jsonl',
        'test': 'data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'infographicsvqa_val': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/val.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'infographicsvqa_test': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/test.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}



class HybridDatasetSequential(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        use_high_res=False,
        bsz=8,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.global_cnt = 0
        self.bsz = bsz

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDatasetSeq(
                        base_image_dir,
                        tokenizer,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        use_high_res,
                        sample_rate[0],
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDatasetSeq(
                        base_image_dir,
                        tokenizer,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        use_high_res,
                        sample_rate[1] if len(sample_rate) > 1 else sample_rate[0],
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        use_high_res,
                        sample_rate[2],
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        use_high_res,
                        sample_rate[3],
                    )
                )
                base_prompt = 'Answer the question using a single word or phrase.'
                vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
                infovqa_prompt = 'Answer the question using a single word or phrase.'
                ai2d_prompt = ''
                # involving additional datasets for VQA training
                # Note: the collection dict is inherited from the eval of internvl series, thus the ds_name contains "val"
                # the data for training are all from training set
                for ds_name in ['vqav2_val', 'okvqa_val', 'textvqa_val', 'vizwiz_val', 'gqa_testdev']:
                    if 'vizwiz' in ds_name:
                        input_prompt = vizwiz_prompt + base_prompt
                    elif 'ai2d' in ds_name:
                        input_prompt = ai2d_prompt
                    elif 'infographicsvqa' in ds_name:
                        input_prompt = infovqa_prompt
                    else:
                        input_prompt = base_prompt

                    dataset = AddVQADataset(
                        train=ds_collections[ds_name]['train'],
                        prompt=input_prompt,
                        use_high_res=use_high_res,
                        base_image_dir="data/ai2diagram/ai2d" if 'ai2diagram' in ds_name else None,
                    )
                    self.all_datasets.append(dataset)
        
        self.seg_sample_pivot = sum([len(ds) for ds in self.all_datasets[:3]]) 
        self.concat_dataset = ConcatDataset(self.all_datasets)
        print(f"HybridDatasetSequential with datasets {self.datasets}, total length {len(self.concat_dataset)}")


    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        self.global_cnt += 1
        if self.global_cnt >= self.bsz:
            self.global_cnt = 0
            rand_idx = random.randint(0, self.seg_sample_pivot-1)
            # print(f"Resample at idx {idx}, seg_sample_pivot {self.seg_sample_pivot}, conversation{self.concat_dataset[rand_idx][2]}")
            assert "[SEG]" in self.concat_dataset[rand_idx][2][0], f"rand_idx{rand_idx}, conversation{self.concat_dataset[rand_idx][2]}"
            return *self.concat_dataset[rand_idx], False
        return *self.concat_dataset[idx], False


class HybridDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        use_high_res=False,
        bsz=8,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.global_cnt = 0
        self.bsz = bsz

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                        use_high_res,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                        use_high_res,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                        use_high_res,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                        use_high_res,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        # Make sure the same dataset is used for the same batch following PSALM
        self.global_cnt += 1
        if self.global_cnt >= self.bsz:
            self.global_cnt = 0
            ind_reproj = np.delete(np.arange(len(self.sample_rate)), -2)
            ind = np.random.choice(ind_reproj, p=self.sample_rate[ind_reproj] / np.sum(self.sample_rate[ind_reproj]))
            data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        val_dataset,
        image_size=1024,
        use_high_res=False,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            self.base_image_dir = os.path.join(
                self.base_image_dir, "refer_seg"
            )
            refer_api = REFER(self.base_image_dir, ds, splitBy) if ds != "grefcoco" else G_REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        self.base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        self.base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.use_high_res = use_high_res
    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    # Handle both single ann_id and list of ann_ids (for grefcoco compatibility)
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
            tmp_image = cv2.imread(image_path)
            tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
            ori_size = tmp_image.shape[:2]
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, tmp_image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.get_conv_template("internvl2_5").copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        resize = image.shape[:2]

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                # Handle both single ann_id and list of ann_ids (for grefcoco compatibility)
                if isinstance(ann_id, list):
                    # For grefcoco, ann_id is a list - process all annotations and merge masks
                    if -1 in ann_id:
                        # Empty annotation case
                        assert len(ann_id) == 1
                        m = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                    else:
                        # Multiple annotations - merge their masks
                        m_final = np.zeros((image_info["height"], image_info["width"])).astype(np.uint8)
                        for ann_id_single in ann_id:
                            ann = annotations[ann_id_single]
                            if not ann["segmentation"]:
                                continue
                            # For list type, check if empty; for dict type, check if it's a valid RLE
                            if isinstance(ann["segmentation"], list) and len(ann["segmentation"]) == 0:
                                continue
                            if isinstance(ann["segmentation"], dict) and ("counts" not in ann["segmentation"] or "size" not in ann["segmentation"]):
                                continue
                            # Check if segmentation is a list (polygon) or dict (RLE)
                            if isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0 and isinstance(ann["segmentation"][0], list):  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                # Handle RLE format (either list of RLE dicts or single RLE dict)
                                if isinstance(ann["segmentation"], dict):
                                    # Single RLE dict
                                    segmentation = ann["segmentation"]
                                    if isinstance(segmentation["counts"], list):
                                        # Convert list format to compressed RLE
                                        rle = mask.frPyObjects([segmentation], image_info["height"], image_info["width"])
                                    else:
                                        rle = [segmentation]
                                        if isinstance(rle[0]["counts"], str):
                                            rle[0]["counts"] = rle[0]["counts"].encode()
                                else:
                                    # List of RLE dicts
                                    rle = ann["segmentation"]
                                    for j in range(len(rle)):
                                        if isinstance(rle[j]["counts"], list):
                                            # Convert list format to compressed RLE
                                            rle[j] = mask.frPyObjects([rle[j]], image_info["height"], image_info["width"])[0]
                                        elif isinstance(rle[j]["counts"], str):
                                            rle[j]["counts"] = rle[j]["counts"].encode()
                            m_single = mask.decode(rle)
                            m_single = np.sum(m_single, axis=2)  # sometimes multiple binary maps
                            m_single = m_single.astype(np.uint8)
                            m_final = np.logical_or(m_final, m_single).astype(np.uint8)
                        m = m_final
                else:
                    # Single annotation case (regular refer datasets)
                    ann = annotations[ann_id]
                    # Check if segmentation is empty or invalid
                    is_empty_segmentation = (
                        not ann["segmentation"] or
                        (isinstance(ann["segmentation"], list) and len(ann["segmentation"]) == 0) or
                        (isinstance(ann["segmentation"], dict) and ("counts" not in ann["segmentation"] or "size" not in ann["segmentation"]))
                    )
                    
                    if is_empty_segmentation and sampled_sents[i] != "":
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    elif is_empty_segmentation:
                        m = np.zeros((image_info["height"], image_info["width"], 1))
                    else:
                        if isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0 and isinstance(ann["segmentation"][0], list):  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"],
                                image_info["height"],
                                image_info["width"],
                            )
                        else:
                            # Handle RLE format (either list of RLE dicts or single RLE dict)
                            if isinstance(ann["segmentation"], dict):
                                # Single RLE dict
                                segmentation = ann["segmentation"]
                                if isinstance(segmentation["counts"], list):
                                    # Convert list format to compressed RLE
                                    rle = mask.frPyObjects([segmentation], image_info["height"], image_info["width"])
                                else:
                                    rle = [segmentation]
                                    if isinstance(rle[0]["counts"], str):
                                        rle[0]["counts"] = rle[0]["counts"].encode()
                            else:
                                # List of RLE dicts
                                rle = ann["segmentation"]
                                for j in range(len(rle)):
                                    if isinstance(rle[j]["counts"], list):
                                        # Convert list format to compressed RLE
                                        rle[j] = mask.frPyObjects([rle[j]], image_info["height"], image_info["width"])[0]
                                    elif isinstance(rle[j]["counts"], str):
                                        rle[j]["counts"] = rle[j]["counts"].encode()
                        m = mask.decode(rle)
                    m = np.sum(
                        m, axis=2
                    )  # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            target_aspect_ratio, 
            inference,
        )
