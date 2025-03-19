"""
Run this after you have the annotation files and all the videos downloaded.

Annotations are from CLIP4Clip: https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip

Script outputs two json files, one for training, one for testing.
"""
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import is_exist


def main(args):

    with open(os.path.join(args.annot_dir, "MSRVTT_data.json"), "rb") as f:
        annot = json.load(f)

    train_df = pd.read_csv(os.path.join(args.annot_dir, "MSRVTT_train.9k.csv"))
    train_vids = train_df["video_id"].tolist()
    val_df = pd.read_csv(os.path.join(args.annot_dir, "MSRVTT_JSFUSION_test.csv"))
    val_vids = val_df["video_id"].tolist()

    train_data = {}
    val_data = {}

    for vid_annot in tqdm(annot["videos"]):
        video_id = vid_annot["video_id"]
        extension = is_exist(args.video_dir, video_id)
        if not extension:
            print(f"{video_id} does not exist.")
        if video_id in train_vids:
            train_data[video_id] = {"extension": extension}
            train_data[video_id]["timestamps"] = [[vid_annot["start time"], vid_annot["end time"]]]
            train_data[video_id]["sentences"] = []
        elif video_id in val_vids:
            val_data[video_id] = {"extension": extension}
            val_data[video_id]["timestamps"] = [[vid_annot["start time"], vid_annot["end time"]]]
            val_data[video_id]["sentences"] = []

    for cap_annot in annot["sentences"]:
        video_id = cap_annot["video_id"]
        if video_id in train_vids:
            train_data[video_id]["sentences"].append(cap_annot["caption"])
        # elif video_id in val_vids:
        #     val_data[video_id]["sentences"].append(cap_annot["caption"])
    
    for video_id in val_vids:
        sentence = val_df.loc[val_df['video_id'] == video_id, 'sentence'].iloc[0] # JS-fusion split
        val_data[video_id]["sentences"].append(sentence)

    Path(os.path.join(args.output_folder)).mkdir(parents=True, exist_ok=True)

    print(f"Number of training videos: {len(train_data)}")
    print(f"Number of validation videos: {len(val_data)}")

    with open(os.path.join(args.output_folder, "train.json"), "w") as f:
        json.dump(train_data, f)

    with open(os.path.join(args.output_folder, "val.json"), "w") as f:
        json.dump(val_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSRVTT-9k annotation preprocessing")
    parser.add_argument('--video_dir', default='/mnt/sda/msrvtt/all_vids', help='path to downloaded videos')
    parser.add_argument('--annot_dir', default='/mnt/sda/msrvtt/clip4clip_annots', help='path to annotation files')
    parser.add_argument('--output_folder', default='annots/msrvtt-jsfusion')
    args = parser.parse_args()

    main(args)