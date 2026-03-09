# SELF1E: Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token


## Notifications

- This code is for the paper [SELF1E: Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token](https://arxiv.org/abs/2603.00000).
- The full training code is released. 

-[ ] TODO: The final version of readme file is not yet completed.
-[ ] TODO: Official weights will be released.

## Abstract

## Highlights



## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training
### Training Data Preparation
The training data consists of 4 types of data:

1. Semantic segmentation datasets: [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip), [Mapillary](https://www.mapillary.com/dataset/vistas), [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup), [PASCAL-Part](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part), [COCO Images](http://images.cocodataset.org/zips/train2017.zip)

    Note: For COCO-Stuff, we use the annotation file stuffthingmaps_trainval2017.zip. We only use the PACO-LVIS part in PACO. COCO Images should be put into the `dataset/coco/` directory.

3. Referring segmentation datasets: [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) 

    Note: the original links of refCOCO series data are down, and we update them with new ones. If the download speed is super slow or unstable, we also provide a [OneDrive link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155154502_link_cuhk_edu_hk/Em5yELVBvfREodKC94nOFLoBLro_LPxsOxNV44PHRWgLcA?e=zQPjsc) to download. **You must also follow the rules that the original datasets require.**

4. Visual Question Answering dataset: [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)

5. Reasoning segmentation dataset: [ReasonSeg](https://github.com/dvlab-research/LISA#dataset)

Download them from the above links, and organize them as follows.

```
├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── mapillary
│   │   ├── config_v2.0.json
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       ├── val
│   │       └── explanatory
│   ├── refer_seg
│   │   ├── images
│   │   |   ├── saiapr_tc-12 
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   └── refcocog
│   └── vlpart
│       ├── paco
│       │   └── annotations
│       └── pascal_part
│           ├── train.json
│           └── VOCdevkit
```

## Dataset
In ReasonSeg, we have collected 1218 images (239 train, 200 val, and 779 test). The training and validation sets can be download from <a href="https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing">**this link**</a>. 

Each image is provided with an annotation JSON file:
```
image_1.jpg, image_1.json
image_2.jpg, image_2.json
...
image_n.jpg, image_n.json
```
Important keys contained in JSON files:
```
- "text": text instructions.
- "is_sentence": whether the text instructions are long sentences.
- "shapes": target polygons.
```

The elements of the "shapes" exhibit two categories, namely **"target"** and **"ignore"**. The former category is indispensable for evaluation, while the latter category denotes the ambiguous region and hence disregarded during the evaluation process. 

We provide a <a href="https://github.com/dvlab-research/LISA/blob/main/utils/data_processing.py">**script**</a> that demonstrates how to process the annotations:
```
python3 utils/data_processing.py
```

Besides, we leveraged GPT-3.5 for rephrasing instructions, so images in the training set may have **more than one instructions (but fewer than six)** in the "text" field. During training, users may randomly select one as the text query to obtain a better model.



## Acknowledgement
-  This work is built upon the [LISA](https://github.com/JIA-Lab-research/LISA) and some of the training settings are borrowed from [PSALM](https://github.com/zamling/PSALM). Thanks for their extraordinary works. 
