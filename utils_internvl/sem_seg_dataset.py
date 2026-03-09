import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from model.internvl3 import conversation as conversation_lib

from .img_loading import load_image

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


MAX_SAMPLES = 30000
def init_mapillary(base_image_dir):
    mapillary_data_root = os.path.join(base_image_dir, "mapillary")
    with open(os.path.join(mapillary_data_root, "config_v2.0.json")) as f:
        mapillary_classes = json.load(f)["labels"]
    mapillary_classes = [x["readable"].lower() for x in mapillary_classes]
    mapillary_classes = np.array(mapillary_classes)
    mapillary_labels = sorted(
        glob.glob(
            os.path.join(mapillary_data_root, "training", "v2.0", "labels", "*.png")
        )
    )
    mapillary_images = [
        x.replace(".png", ".jpg").replace("v2.0/labels", "images")
        for x in mapillary_labels
    ]
    print("mapillary: ", len(mapillary_images))
    return mapillary_classes, mapillary_images, mapillary_labels


def init_ade20k(base_image_dir):
    with open("utils/ade20k_classes.json", "r") as f:
        ade20k_classes = json.load(f)
    ade20k_classes = np.array(ade20k_classes)
    image_ids = sorted(
        os.listdir(os.path.join(base_image_dir, "ade20k/images", "training"))
    )
    ade20k_image_ids = []
    for x in image_ids:
        if x.endswith(".jpg"):
            ade20k_image_ids.append(x[:-4])
    ade20k_images = []
    for image_id in ade20k_image_ids:  # self.descriptions:
        ade20k_images.append(
            os.path.join(
                base_image_dir,
                "ade20k",
                "images",
                "training",
                "{}.jpg".format(image_id),
            )
        )
    ade20k_labels = [
        x.replace(".jpg", ".png").replace("images", "annotations")
        for x in ade20k_images
    ]
    print("ade20k: ", len(ade20k_images))
    return ade20k_classes, ade20k_images, ade20k_labels


def init_cocostuff(base_image_dir):
    cocostuff_classes = []
    with open("utils/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)
    cocostuff_images = []

    cocostuff_labels = glob.glob(
        os.path.join(base_image_dir, "cocostuff", "train2017", "*.png")
    )
    if len(cocostuff_labels) > MAX_SAMPLES:
        cocostuff_labels = random.sample(cocostuff_labels, MAX_SAMPLES)
    cocostuff_images = [
        x.replace(".png", ".jpg").replace("cocostuff", "coco") for x in cocostuff_labels
    ]

    print("cocostuff: ", len(cocostuff_images))
    return cocostuff_classes, cocostuff_images, cocostuff_labels


def init_paco_lvis(base_image_dir):
    coco_api_paco_lvis = COCO(
        os.path.join(
            base_image_dir, "vlpart", "paco", "annotations", "paco_lvis_v1_train.json"
        )
    )
    all_classes = coco_api_paco_lvis.loadCats(coco_api_paco_lvis.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name
    img_ids = coco_api_paco_lvis.getImgIds()
    if len(img_ids) > MAX_SAMPLES:
        img_ids = random.sample(img_ids, MAX_SAMPLES)
    print("paco_lvis: ", len(img_ids))
    return class_map_paco_lvis, img_ids, coco_api_paco_lvis


def init_pascal_part(base_image_dir):
    coco_api_pascal_part = COCO(
        os.path.join(base_image_dir, "vlpart", "pascal_part", "train.json")
    )
    all_classes = coco_api_pascal_part.loadCats(coco_api_pascal_part.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = coco_api_pascal_part.getImgIds()
    print("pascal_part: ", len(img_ids))
    return class_map_pascal_part, img_ids, coco_api_pascal_part

def init_coco_rem(base_image_dir):
    coco_api_coco_rem = COCO(
        os.path.join(base_image_dir, "coco_rem", "instances_trainrem.json")
    )
    all_classes = coco_api_coco_rem.loadCats(coco_api_coco_rem.getCatIds())
    class_map_coco_rem = {}
    for cat in all_classes:
        class_map_coco_rem[cat["id"]] = cat["name"]
    img_ids = coco_api_coco_rem.getImgIds()
    print("coco_rem: ", len(img_ids))
    return class_map_coco_rem, img_ids, coco_api_coco_rem


class SemSegDataset(torch.utils.data.Dataset):
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
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        use_high_res=False,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.use_high_res = use_high_res
        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes

        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

    def __len__(self):
        return len(self.sem_seg_datas)

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]

        if ds in ["paco_lvis", "pascal_part", "coco_rem"]:
            class_map = self.data2classes[ds]
            img_ids, coco_api = self.data2list[ds]
            idx = random.randint(0, len(img_ids) - 1)
            img_id = img_ids[idx]
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            elif ds == "coco_rem":
                image_path = os.path.join(self.base_image_dir, "coco/train2017", file_name)
            image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)
            if len(anns) == 0:
                return self.__getitem__(0)
            if len(anns) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    if random.random() < 0.5:
                        name = obj + " " + part
                    else:
                        name = "the {} of the {}".format(part, obj)
                else:
                    name = sampled_cls
                sampled_classes.append(name)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            image, labels = self.data2list[ds]
            idx = random.randint(0, len(image) - 1)
            image_path = image[idx]
            label_path = labels[idx]
            label = Image.open(label_path)
            label = np.array(label)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)
            if len(unique_label) == 0:
                return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes

        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))

            if ds in ["paco_lvis", "pascal_part"]:
                continue

            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids.append(class_id)

        conversations = []
        conv = conversation_lib.get_conv_template("internvl2_5").copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        if ds in ["paco_lvis", "pascal_part"]:
            masks = []
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(e)
                    return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)
        assert masks.shape[0]==len(conversations), f"masks{masks.shape[0]}, conversations{len(conversations)}"
        return (
            image_path,
            image,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            target_aspect_ratio
        )

class SemSegDatasetSeq(torch.utils.data.Dataset):
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        use_high_res=False,
        sample_rate=1.0,
    ):
        self.exclude_val = exclude_val
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.use_high_res = use_high_res
        
        # self.data_cache 存储每个数据集的原始加载结果
        self.data_cache = {}
        # self.samples 是核心改动，一个包含所有样本元信息的扁平化列表
        self.samples = []

        sem_seg_datas = sem_seg_data.split("||")
        print("Start loading and combining datasets...")
        for ds in sem_seg_datas:
            # 加载每个数据集的原始数据
            # 注意：原始代码中的 eval 是动态调用函数，这里保持不变
            raw_data = eval("init_{}".format(ds))(base_image_dir)
            self.data_cache[ds] = raw_data

            # 根据数据集类型，填充 self.samples 列表
            # 类型1: 基于COCO API的数据集 (paco_lvis, pascal_part, etc.)
            if ds in ["paco_lvis", "pascal_part", "coco_rem"]:
                class_map, img_ids, coco_api = raw_data
                for img_id in img_ids:
                    # 每个样本的信息包含其所属数据集和图片ID
                    self.samples.append({"dataset": ds, "image_id": img_id})
            
            # 类型2: 基于文件列表的数据集 (ade20k, cocostuff, etc.)
            elif ds in ["ade20k", "cocostuff", "mapillary"]:
                classes, images, labels = raw_data
                for i in range(len(images)):
                    # 每个样本的信息包含其所属数据集和在该数据集中的索引
                    self.samples.append({"dataset": ds, "sample_idx": i})

        if "cocostuff" in sem_seg_datas:
            cocostuff_classes = self.data_cache["cocostuff"][0]
            self.cocostuff_class2index = {c: i for i, c in enumerate(cocostuff_classes)}

        # copy sample-rate times of self.samples
        self.samples = self.samples * int(sample_rate)
            
        print(f"All datasets have been loaded and combined. Total number of samples: {len(self.samples)}")

    def __len__(self):
        # 返回拼接后所有样本的总数
        return len(self.samples)

    def __getitem__(self, idx):
        # **核心改动**: 直接通过索引 `idx` 从总样本列表中获取样本信息
        sample_info = self.samples[idx]
        ds = sample_info["dataset"]

        # ----- 以下是根据获取到的 ds 和 sample_info 来加载数据的逻辑 -----
        
        if ds in ["paco_lvis", "pascal_part", "coco_rem"]:
            # 从缓存中获取该数据集的原始数据
            class_map, img_ids, coco_api = self.data_cache[ds]
            # 从 sample_info 中获取 image_id，不再随机选择
            img_id = sample_info["image_id"]
            
            image_info = coco_api.loadImgs([img_id])[0]
            file_name = image_info["file_name"]
            
            # (文件路径处理逻辑保持不变)
            if ds == "pascal_part":
                file_name = os.path.join(
                    "VOCdevkit", "VOC2010", "JPEGImages", file_name
                )
                image_path = os.path.join(self.base_image_dir, "vlpart", ds, file_name)
            elif ds == "paco_lvis":
                image_path = os.path.join(self.base_image_dir, "coco", file_name)
            elif ds == "coco_rem":
                image_path = os.path.join(self.base_image_dir, "coco/train2017", file_name)
                
            image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
            resize = image.shape[:2]
            annIds = coco_api.getAnnIds(imgIds=image_info["id"])
            anns = coco_api.loadAnns(annIds)

            if len(anns) == 0:
                return self.__getitem__((idx + 1) % len(self)) # 避免无限递归，取下一个样本

            # (采样和文本生成逻辑保持不变)
            if len(anns) >= self.num_classes_per_sample:
                sampled_anns = np.random.choice(
                    anns, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_anns = anns
            
            sampled_classes = []
            for ann in sampled_anns:
                sampled_cls = class_map[ann["category_id"]]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
                else:
                    name = sampled_cls
                sampled_classes.append(name)

        elif ds in ["ade20k", "cocostuff", "mapillary"]:
            # 从缓存中获取该数据集的原始数据
            classes, images, labels = self.data_cache[ds]
            # 从 sample_info 中获取样本索引，不再随机选择
            sample_idx = sample_info["sample_idx"]
            
            image_path = images[sample_idx]
            label_path = labels[sample_idx]
            label = Image.open(label_path)
            label = np.array(label)

            # (标签预处理逻辑保持不变)
            if ds == "ade20k":
                label[label == 0] = 255
                label -= 1
                label[label == 254] = 255
            elif ds == "cocostuff":
                for c, i in self.cocostuff_class2index.items():
                    if "-" in c:
                        label[label == i] = 255
            
            image, target_aspect_ratio = load_image(image_path, max_num=4 if self.use_high_res else 1)
            resize = image.shape[:2]
            unique_label = np.unique(label).tolist()
            if 255 in unique_label:
                unique_label.remove(255)

            if len(unique_label) == 0:
                return self.__getitem__((idx + 1) % len(self)) # 避免无限递归，取下一个样本

            # (采样和文本生成逻辑保持不变)
            all_classes_in_image = [classes[class_id] for class_id in unique_label]
            if len(all_classes_in_image) >= self.num_classes_per_sample:
                sampled_classes = np.random.choice(
                    all_classes_in_image, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = all_classes_in_image

        # ----- 以下是所有数据集类型共享的处理逻辑，保持不变 -----
        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

            if ds in ["paco_lvis", "pascal_part", "coco_rem"]:
                continue
            
            # 需要从缓存中获取类别列表
            class_list = self.data_cache[ds][0].tolist()
            class_id = class_list.index(sampled_cls)
            class_ids.append(class_id)

        conversations = []
        conv = conversation_lib.get_conv_template("internvl2_5").copy()
        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        if ds in ["paco_lvis", "pascal_part", "coco_rem"]:
            masks = []
            coco_api = self.data_cache[ds][2] # 重新获取 coco_api
            for ann in sampled_anns:
                try:
                    masks.append(coco_api.annToMask(ann))
                except Exception as e:
                    print(f"Error creating mask for ann {ann['id']} in image {image_path}: {e}")
                    return self.__getitem__((idx + 1) % len(self))
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        else:
            label = torch.from_numpy(label).long()
            masks = []
            for class_id in class_ids:
                masks.append(label == class_id)
            masks = torch.stack(masks, dim=0)
        
        assert masks.shape[0]==len(conversations), f"masks{masks.shape[0]}, conversations{len(conversations)}"
        
        return (
            image_path,
            image,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            target_aspect_ratio
        )