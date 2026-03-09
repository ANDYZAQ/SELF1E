import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from .img_loading import load_image

from model.internvl3 import conversation as conversation_lib

from .utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
    return source


MAX_SAMPLES = 100000
class VQADataset(torch.utils.data.Dataset):
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
        vqa_data="llava_v1_5_mix665k",
        use_high_res=False,
        sample_rate=1.0,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.use_high_res = use_high_res
        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        # self.vqa_image_root = base_image_dir
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data
        # if len(self.vqa_data) > MAX_SAMPLES:
        #     self.vqa_data = random.sample(self.vqa_data, MAX_SAMPLES)
        # copy sample-rate times of self.vqa_data
        self.vqa_data = self.vqa_data * int(sample_rate)

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return len(self.vqa_data)

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
        resize = image.shape[:2]

        conv = conversation_lib.get_conv_template("internvl2_5").copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source
        )
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        ori_size = resize
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

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


class AddVQADataset(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(self, train, prompt, base_image_dir=None, use_high_res=False, sample_rate=1.0):
        self.prompt = prompt
        self.use_high_res = use_high_res
        self.train = open(train).readlines()
        self.base_image_dir = base_image_dir
        if len(self.train) > MAX_SAMPLES:
            self.train = random.sample(self.train, MAX_SAMPLES)
        self.train = random.sample(self.train, int(len(self.train) * sample_rate))
        print(f"AddVQADataset with {len(self.train)} samples, use_high_res: {self.use_high_res}")

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        data = json.loads(self.train[idx].strip())
        image_path = data['image']
        if self.base_image_dir is not None:
            image_path = os.path.join(self.base_image_dir, image_path)
        question = data['question']
        question_id = data['question_id']
        annotation = data.get('answer', None)
        question = '<image>\n' + question + ' ' + self.prompt

        # load and resize image
        image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
        resize = image.shape[:2]

        # conversations structure
        conv = conversation_lib.get_conv_template("internvl2_5").copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        if annotation is not None:
            conv.append_message(conv.roles[1], annotation)
        conversations = [conv.get_prompt()]

        questions = conversations
        sampled_classes = conversations

        ori_size = resize
        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

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