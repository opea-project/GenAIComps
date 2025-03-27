# Dataset

## Dataset for CLIP

### Caltech101

- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`.

The directory structure should look like

```
$DATA/
|-- caltech-101/
|   |-- 101_ObjectCategories/
|   | split_zhou_Caltech101.json
```

### mini-imagenet

- Create a folder named `mini-imagenet/` under `$DATA`.
- Download the dataset from the [mini-imagnet](https://yaoyaoliu.web.illinois.edu/projects/mtl/download/) and extract the training and validation sets to `$DATA/mini-imagenet`.
- Download the `classnames.txt` to `$DATA/mini-imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

The directory structure should look like

```
$DATA/
|–– mini-imagenet/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
|   |–– test/
|   |-- classnames.txt
```

### MSCOCO2014

- Create a folder named `mscoco2014/` under `$DATA`.
- Download the dataset from the [MSCOCO](https://cocodataset.org/#download) and extract the training and validation sets to `$DATA/mscoco2014`.
- download json file from `https://www.kaggle.com/datasets/wangjilong/dataset-json/` data/mscococ2014/\*.json to `$DATA/mscoco2014`

The directory structure should look like

```
$DATA/
|–– mscoco2014/
|   |–– train2014/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val2014/
|   |-- captions_train.json
|   |-- coco_karpathy_test.json
|   |-- coco_karpathy_val.json
```

### Flickr

- Create a folder name `flickr/` under `$DATA`.
- Download the dataset form the [Kaggle](https://www.kaggle.com/datasets/eeshawn/flickr30k/data)
- download json file from `https://www.kaggle.com/datasets/wangjilong/dataset-json/` data/flickr/\*.json to `$DATA/flickr`

```
$DATA/
|–– flickr/
|   |–– flickr30k-images/
|   |   |-- *.jpg
|   |-- flickr30k_train.json.json
|   |-- flickr30k_val.json.json
|   |-- flickr30k_test.json.json
```

### Flickr5k

- Create a folder name `flickr5k/` under `$DATA`.
- Download the dataset form the [Kaggle](https://www.kaggle.com/datasets/wangjilong/self-data/code)
- download json file from `https://www.kaggle.com/datasets/wangjilong/dataset-json/` data/flickr5k/\*.json to `$DATA/flickr5k`

```
$DATA/
|–– flickr5k/
|   |–– flickr5k-images/
|   |   |-- *.jpg
|   |-- flickr5k_train.json.json
|   |-- flickr5k_val.json.json
|   |-- flickr5k_test.json.json
```

## Dataset for AdaCLIP

### MSRVTT

The videos are shared by [Frozen in Time](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt):

```
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

### DiDeMo

The videos can be downloaded from [LisaAnne/LocalizingMoments](https://github.com/LisaAnne/LocalizingMoments).

### ActivityNet

Download the videos from the [official website](http://activity-net.org/download.html). The authors have made the videos available on Google and Baidu drives.

## Preprocessing

### Frame Extraction

Run `src/adaclip_finetune/utils/frame_extraction.py` after having downloaded the dataset videos and annotations from the website. Make sure that all the videos are in the same directory (no sub-directories allowed).

```
python src/adaclip_finetune/utils/frame_extraction.pyy /path/to/videos /path/to/frames --parallel
```

Subsequently, update the `frames_dir` parameter in the config files `configs/[dataset].json`.

### Annotation Preprocessing

If the videos downloaded differ from the set used in the paper, run `annot_preprocess/{dataset}_preprocess.py` to generate train/test splits used by the dataloader. Splits used in the paper can be found in `annots/`.

To obtain the annotation files used to generate the splits, please download them from the following links:

- MSRVTT annotations are from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip):

```
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```

- ActivityNet annotations are from the [project page](https://cs.stanford.edu/people/ranjaykrishna/densevid/) of ActivityNet Captions:

```
wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
```

- DiDeMo annotations have two components: annotations from the [original author](https://github.com/LisaAnne/LocalizingMoments/tree/master/data) and the split used by [Collaborative Experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/didemo).
