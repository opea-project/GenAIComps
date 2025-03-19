"""
Run this after you have the annotation files and all the videos downloaded.

Annotations are from Localization Moments in Video with Natural Language: https://github.com/LisaAnne/LocalizingMoments/tree/master/data
and Collaborative Experts: https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/didemo

Script outputs three json files, for train, val, and test.
"""
import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from utils import is_exist


def main(args, subset):

    with open(os.path.join(args.annot_dir, f"{subset}_list.txt"), "r") as f:
        video_list = f.read().splitlines()

    with open(os.path.join(args.annot_dir, f"{subset}_data.json"), "rb") as f:
        annot = json.load(f)

    output_data = {}

    for data in tqdm(annot):
        if data["video"] not in video_list:
            # print(f"{data['video']} does not belong in CE split.")
            continue
        extension = is_exist(args.video_dir, data["video"])
        if not extension:
            print(f"{data['video']} does not exist.")
            continue
        if not data["description"].endswith("."):
            data["description"] = data["description"] + "."
        if data["video"] in output_data:
            output_data[data["video"]]["sentences"].append(data["description"])
        else:
            output_data[data["video"]] = {"extension": extension}
            output_data[data["video"]]["sentences"] = [data["description"]]

    Path(os.path.join(args.output_folder)).mkdir(parents=True, exist_ok=True)

    print(f"Number of {subset} videos: {len(output_data)}")

    with open(os.path.join(args.output_folder, f"{subset}.json"), "w") as f:
        json.dump(output_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiDeMo annotation preprocessing")
    parser.add_argument('--video_dir', default='/mnt/sdb/angela/didemo/videos', help='path to downloaded videos')
    parser.add_argument('--annot_dir', default='/mnt/sdb/angela/didemo/annots', help='path to annotation files')
    parser.add_argument('--output_folder', default='annots/didemo')
    args = parser.parse_args()

    main(args, "train")
    main(args, "val")
    main(args, "test")