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
|   |-- flickr30k_train.json
|   |-- flickr30k_val.json
|   |-- flickr30k_test.json
```

### FlickrCN

- Create a folder name `flickrcn/` under `$DATA`.
- Download the dataset form the [Kaggle](https://www.kaggle.com/datasets/eeshawn/flickr30k/data)
- download json file from `https://huggingface.co/datasets/OFA-Sys/chinese-clip-eval/resolve/main/Flickr30k-CN.zip` Flickr30k-CN.zip\Flickr30k-CN/\*.jsonl to `$DATA/flickrcn`

```
$DATA/
|–– flickrcn/
|   |–– flickr30k-images/
|   |   |-- *.jpg
|   |-- train_texts.jsonl
|   |-- val_texts.jsonl
|   |-- test_texts.jsonl
```
- Run `generate_flickr30k_cn_json.py --base_dir $DATA/flickrcn/` to generate usable json file
```
$DATA/
|–– flickrcn/
|   |–– flickr30k-images/
|   |   |-- *.jpg
|   |-- flickr30k_cn_train.json
|   |-- flickr30k_cn_val.json
|   |-- flickr30k_cn_test.json
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

## Dataset for Qwen2-VL Finetune

### ActivityNet-QA

Please follow https://github.com/MILVLG/activitynet-qa/tree/master to download and seperata train/val dataset

Then use below python generate_llama_json_limit_frames.py file to generate our train and test dataset:
python generate_llama_json_limit_frames.py -name val_q -type val -n 500 -seconds 20

generate_llama_json_limit_frames.py

```python
import json
import os
import argparse
import ffmpeg

# Define the path to the directory where the video files are stored
video_directory = "where to find dataset"


def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        return float(video_stream["duration"])
    except Exception as e:
        print(f"Error getting duration for video {video_path}: {e}")
        return 0


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate LLaMA JSON")
    parser.add_argument("-name", type=str, default="train_q_3000", help="Number of questions to process")
    parser.add_argument("-type", type=str, default="train", help="data type")
    parser.add_argument("-fps", type=float, default=0.2, help="data type")
    parser.add_argument("-n", type=int, default=250, help="data type")
    parser.add_argument("-seconds", type=int, default=20, help="minimum video duration in seconds")
    args = parser.parse_args()
    fps = args.fps
    basic_seconds = args.seconds
    question_json = "../activitynet-qa/dataset/{}.json".format(args.name)
    answer_json = "../activitynet-qa/dataset/{}_a.json".format(args.type)
    combine_json = "../data/activitynet_qa_{}_{}_limit_{}s.json".format(args.type, args.n, basic_seconds)
    print("combine_json:", combine_json)

    # Supported video file extensions
    video_extensions = (".mp4", ".mkv", "webm")

    # Load the questions and answers JSON files
    with open(question_json, "r") as question_file:
        questions = json.load(question_file)

    with open(answer_json, "r") as answer_file:
        answers = json.load(answer_file)

    # Create a dictionary to map question_id to answer for quick lookup
    answer_lookup = {answer["question_id"]: answer for answer in answers}

    combined_data = []
    len_pairs = len(questions)
    # Process each question and look for a corresponding answer
    for question in questions:
        question_id = question["question_id"]
        if question_id in answer_lookup:
            answer = answer_lookup[question_id]

            # Extract the video name typically between 'v_' and the second underscore or end
            video_name_without_path = ("_").join(question_id.split("_")[:-1])
            # Search for the video file that matches the extracted name
            video_path = None
            find_flag = False
            # Walk through the directory to find matching video files
            for root, dirs, files in os.walk(video_directory):
                for file in files:
                    if file.startswith(video_name_without_path) and file.endswith(video_extensions):
                        video_path = os.path.join(root, file)
                        find_flag = True
                        break
                if video_path:
                    break
            if not find_flag:
                print("!!not find:", video_name_without_path)
            if video_path:
                video_duration = get_video_duration(video_path)
                if video_duration > basic_seconds:
                    combined_entry = {
                        "messages": [
                            {"content": f"<video>{question['question']}?", "role": "user"},
                            {"content": answer["answer"], "role": "assistant"},
                        ],
                        "videos": [video_path],
                    }
                    combined_data.append(combined_entry)
                    if len(combined_data) % 100 == 0:
                        print(f"Processed {len(combined_data)} entries")
                    if len(combined_data) >= args.n:
                        break
                else:
                    print("video_duration < basic_seconds", video_duration, video_path)
    # Write the combined data to the output JSON file
    with open(combine_json, "w") as combine_file:
        json.dump(combined_data, combine_file, indent=4)
```

## Update dataset_info.json

### dataset_info.json

```json
{
  "caltech101": {
    "file_name": "caltech101.json"
  },
  "ActivityNet": {
    "file_name": "ActivityNet.json"
  },
  "activitynet_qa_2000_limit_20s": {
    "file_name": "activitynet_qa_2000_limit_20s.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "videos": "videos"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### caltech101.json

```json
[]
```

### ActivityNet.json

```json
[]
```

### activitynet_qa_2000_limit_20s.json

Generate by generate_llama_json_limit_frames.py
