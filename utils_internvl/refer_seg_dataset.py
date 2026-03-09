import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from .img_loading import load_image

from model.internvl3 import conversation as conversation_lib

from .grefer import G_REFER
from .refer import REFER
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


class ReferSegDataset(torch.utils.data.Dataset):
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
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        use_high_res=False,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.use_high_res = use_high_res
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_seg_ds_list = refer_seg_data.split(
            "||"
        )  # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_seg_data = {}
        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

            print(
                "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                    ds,
                    splitBy,
                    len(refer_seg_ds["images"]),
                    len(refer_seg_ds["annotations"]),
                )
            )

            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
        resize = image.shape[:2]

        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.get_conv_template("internvl2_5").copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if not ann["segmentation"]:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        elif isinstance(ann["segmentation"], list) and len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        elif isinstance(ann["segmentation"], list) and isinstance(ann["segmentation"][0], list):  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"],
                                image_info["height"],
                                image_info["width"],
                            )
                            m = mask.decode(rle)
                            m = np.sum(m, axis=2).astype(np.uint8)
                        else:  # RLE format (dict or list of dicts)
                            rle = ann["segmentation"]
                            if isinstance(rle, dict):
                                rle = [rle]
                            for i in range(len(rle)):
                                if isinstance(rle[i]["counts"], list):
                                    # Convert list format to compressed RLE
                                    rle[i] = mask.frPyObjects([rle[i]], image_info["height"], image_info["width"])[0]
                                elif isinstance(rle[i]["counts"], str):
                                    rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(m, axis=2).astype(np.uint8)
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if not ann["segmentation"]:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue
            elif isinstance(ann["segmentation"], list) and len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue
            elif isinstance(ann["segmentation"], list) and isinstance(ann["segmentation"][0], list):  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:  # RLE format (dict or list of dicts)
                rle = ann["segmentation"]
                if isinstance(rle, dict):
                    rle = [rle]
                for i in range(len(rle)):
                    if isinstance(rle[i]["counts"], list):
                        # Convert list format to compressed RLE
                        rle[i] = mask.frPyObjects([rle[i]], image_info["height"], image_info["width"])[0]
                    elif isinstance(rle[i]["counts"], str):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2).astype(np.uint8)  # sometimes there are multiple binary map (corresponding to multiple segs)
            masks.append(m)

        assert len(masks)==len(conversations), f"masks{len(masks)}, conversations{len(conversations)}"
        masks = np.stack(masks, axis=0)

        # if ds == 'grefcoco' and flag:
        #     import shutil
        #     image_name = image_path.split("/")[-1]
        #     save_dir = os.path.join("/group/30042/xlai/LISA_refactor_final/debug", image_name.split(".")[0])
        #     os.makedirs(save_dir, exist_ok=True)
        #     shutil.copy(image_path, save_dir)
        #     for i in range(masks.shape[0]):
        #         cv2.imwrite(os.path.join(save_dir, "{}_{}_{}.jpg".format(image_name, i, sampled_classes[i])), masks[i].astype(np.int32) * 100)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            target_aspect_ratio,
        )


# 假设这些依赖项在您的项目中已经定义好了
# from .refer import REFER, G_REFER
# from .utils.conversation import conversation_lib
# from .utils.data_processing import load_image
# SHORT_QUESTION_LIST = [...]
# ANSWER_LIST = [...]

class ReferSegDatasetSeq(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        use_high_res=False,
        sample_rate=1.0,
    ):
        self.exclude_val = exclude_val
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.use_high_res = use_high_res
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        refer_seg_ds_list = refer_seg_data.split("||")
        
        # self.refer_seg_data 用于存储每个子数据集加载后的原始数据 (API对象、标注等)
        self.refer_seg_data = {} 
        # self.samples 是一个扁平化的列表，包含所有数据集中所有样本的元信息
        self.samples = []

        print("Start loading datasets...")
        for ds in refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            if ds == "grefcoco":
                refer_api = G_REFER(DATA_DIR, ds, splitBy)
            else:
                refer_api = REFER(DATA_DIR, ds, splitBy)
            
            ref_ids_train = refer_api.getRefIds(split="train")
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

            # 加载图片信息，并创建一个从 image_id 到 image_info 的映射字典，方便后续索引
            loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
            image_id_to_info = {}
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/saiapr_tc-12", item["file_name"]
                    )
                else:
                    item["file_name"] = os.path.join(
                        DATA_DIR, "images/mscoco/images/train2014", item["file_name"]
                    )
                image_id_to_info[item['id']] = item

            # 创建一个从 image_id 到其所有指代表达 (references) 的映射字典
            img2refs = {}
            for ref in refs_train:
                image_id = ref["image_id"]
                if image_id not in img2refs:
                    img2refs[image_id] = []
                img2refs[image_id].append(ref)
            
            # 存储该数据集需要的所有信息
            self.refer_seg_data[ds] = {
                "image_id_to_info": image_id_to_info,
                "annotations": refer_api.Anns,
                "img2refs": img2refs
            }

            print(
                f"dataset {ds} ({splitBy} split) has {len(image_id_to_info)} images "
                f"and {len(ref_ids_train)} referring expressions."
            )

            for image_id in images_ids_train:
                if image_id in img2refs and len(img2refs[image_id]) > 0:
                     self.samples.append({'ds': ds, 'image_id': image_id})
        # copy sample-rate times of self.samples
        self.samples = self.samples * int(sample_rate)

        print(f"All datasets have been loaded and combined. Total number of samples: {len(self.samples)}")

    def __len__(self):
        # 返回所有数据集样本总数
        return len(self.samples)

    def __getitem__(self, idx):
        # **核心改动**: 直接通过索引 `idx` 从拼接好的样本列表中获取样本信息
        # 不再随机选择数据集和图片
        sample_info = self.samples[idx]
        ds = sample_info['ds']
        image_id = sample_info['image_id']

        # 从 self.refer_seg_data 中获取该样本对应的具体数据
        refer_seg_ds = self.refer_seg_data[ds]
        image_info = refer_seg_ds["image_id_to_info"][image_id]
        annotations = refer_seg_ds["annotations"]
        refs = refer_seg_ds["img2refs"][image_id]
        
        image_path = image_info["file_name"]
        
        # 兜底逻辑，以防万一某个 image_id 没有对应的 ref (在新的初始化逻辑下基本不会发生)
        if len(refs) == 0:
            return self.__getitem__(0)

        # ----- 以下的数据处理逻辑与您的原始代码保持一致 -----
        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])

        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
            
        sampled_sents = [sents[ind] for ind in sampled_inds]
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        
        image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
        resize = image.shape[:2]

        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.get_conv_template("internvl2_5").copy()
        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if not ann["segmentation"]:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        elif isinstance(ann["segmentation"], list) and len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        elif isinstance(ann["segmentation"], list) and isinstance(ann["segmentation"][0], list):  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"],
                                image_info["height"],
                                image_info["width"],
                            )
                            m = mask.decode(rle)
                            m = np.sum(m, axis=2).astype(np.uint8)
                        else:  # RLE format (dict or list of dicts)
                            rle = ann["segmentation"]
                            if isinstance(rle, dict):
                                rle = [rle]
                            for i in range(len(rle)):
                                if isinstance(rle[i]["counts"], list):
                                    # Convert list format to compressed RLE
                                    rle[i] = mask.frPyObjects([rle[i]], image_info["height"], image_info["width"])[0]
                                elif isinstance(rle[i]["counts"], str):
                                    rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(m, axis=2).astype(np.uint8)
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if not ann["segmentation"]:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue
            elif isinstance(ann["segmentation"], list) and len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue
            elif isinstance(ann["segmentation"], list) and isinstance(ann["segmentation"][0], list):  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:  # RLE format (dict or list of dicts)
                rle = ann["segmentation"]
                if isinstance(rle, dict):
                    rle = [rle]
                for i in range(len(rle)):
                    if isinstance(rle[i]["counts"], list):
                        # Convert list format to compressed RLE
                        rle[i] = mask.frPyObjects([rle[i]], image_info["height"], image_info["width"])[0]
                    elif isinstance(rle[i]["counts"], str):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2).astype(np.uint8)  # sometimes there are multiple binary map (corresponding to multiple segs)
            masks.append(m)
            
        assert len(masks)==len(conversations), f"masks{len(masks)}, conversations{len(conversations)}"
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            target_aspect_ratio,
        )