"""
Run this after you have the annotation files and all the videos downloaded.

Script outputs two json files, one for training, one for testing.
"""
import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from utils import is_exist


def main(args, subset):

    with open(os.path.join(args.annot_dir, f"{subset}.json"), "rb") as f:
        annot = json.load(f)

    output_data = {}

    for video_id, data in tqdm(annot.items()):
        extension = is_exist(args.video_dir, video_id)
        if not extension:
            print(f"{video_id} does not exist.")
            continue
        output_data[video_id] = {"extension": extension}
        output_data[video_id]["sentences"] = data["sentences"]
        output_data[video_id]["timestamps"] = data["timestamps"]
        output_data[video_id]["duration"] = data["duration"]

    Path(os.path.join(args.output_folder)).mkdir(parents=True, exist_ok=True)

    print(f"Number of {subset} videos: {len(output_data)}")

    if subset == "val_1":
        subset = "val"

    with open(os.path.join(args.output_folder, f"{subset}.json"), "w") as f:
        json.dump(output_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ActivityNet annotation preprocessing")
    parser.add_argument('--video_dir', default='/mnt/sda/activitynet/videos', help='path to downloaded videos')
    parser.add_argument('--annot_dir', default='/mnt/sda/activitynet/annots', help='path to annotation files')
    parser.add_argument('--output_folder', default='annots/activitynet')
    args = parser.parse_args()

    main(args, "train")
    main(args, "val_1")