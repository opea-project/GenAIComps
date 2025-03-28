# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import re

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import mkdir_if_missing, read_json

from .oxford_pets import OxfordPets


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


@DATASET_REGISTRY.register()
class ITC_Mscoco(DatasetBase):

    dataset_dir = "mscoco2014"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
                val = preprocessed["val"]
        else:
            train_file = os.path.join(self.dataset_dir, "captions_train.json")
            val_file = os.path.join(self.dataset_dir, "coco_karpathy_val.json")
            test_file = os.path.join(self.dataset_dir, "coco_karpathy_test.json")

            train = self.read_train_data(train_file)
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_valtest_data(test_file)
            # print(test)
            val = self.read_valtest_data(val_file)
            # print(val)

            preprocessed = {"train": train, "test": test, "val": val}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
        print("len", len(self._classnames))
        # 29000
        # 144481
        # self._classnames = ['house finch', 'American robin', 'triceratops', 'green mamba', 'harvestman', 'toucan', 'goose', 'jellyfish', 'nematode', 'red king crab', 'dugong', 'Treeing Walker Coonhound', 'Ibizan Hound', 'Saluki', 'Golden Retriever', 'Gordon Setter', 'Komondor', 'Boxer', 'Tibetan Mastiff', 'French Bulldog', 'Alaskan Malamute', 'Dalmatian', 'Newfoundland dog', 'Miniature Poodle', 'Alaskan tundra wolf', 'African wild dog', 'Arctic fox', 'lion', 'meerkat', 'ladybug', 'rhinoceros beetle', 'ant', 'black-footed ferret', 'three-toed sloth', 'rock beauty fish', 'aircraft carrier', 'trash can', 'barrel', 'beer bottle', 'bookstore', 'cannon', 'carousel', 'cardboard box / carton', 'catamaran', 'bell or wind chime', 'clogs', 'cocktail shaker', 'combination lock', 'crate', 'cuirass', 'dishcloth', 'dome', 'electric guitar', 'filing cabinet', 'fire screen', 'frying pan', 'garbage truck', 'hair clip', 'holster', 'gymnastic horizontal bar', 'hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile', 'mixing bowl', 'oboe', 'pipe organ', 'parallel bars', 'pencil case', 'photocopier', 'poncho', 'prayer rug', 'fishing casting reel', 'school bus', 'scoreboard', 'slot machine', 'snorkel', 'solar thermal collector', 'spider web', 'stage', 'tank', 'front curtain', 'tile roof', 'tobacco shop', 'unicycle', 'upright piano', 'vase', 'wok', 'split-rail fence', 'sailboat', 'traffic or street sign', 'consomme', 'trifle', 'hot dog', 'orange', 'cliff', 'coral reef', 'bolete', 'corn cob']
        # self._lab2cname = {73: 'fishing casting reel', 1: 'American robin', 18: 'Tibetan Mastiff', 19: 'French Bulldog', 25: 'African wild dog', 99: 'corn cob', 43: 'catamaran', 42: 'cardboard box / carton', 49: 'cuirass', 81: 'tank', 55: 'frying pan', 90: 'sailboat', 72: 'prayer rug', 8: 'nematode', 22: 'Newfoundland dog', 30: 'rhinoceros beetle', 68: 'parallel bars', 95: 'orange', 53: 'filing cabinet', 52: 'electric guitar', 2: 'triceratops', 14: 'Golden Retriever', 59: 'gymnastic horizontal bar', 82: 'front curtain', 5: 'toucan', 9: 'red king crab', 79: 'spider web', 63: 'miniskirt', 87: 'vase', 11: 'Treeing Walker Coonhound', 7: 'jellyfish', 86: 'upright piano', 56: 'garbage truck', 46: 'cocktail shaker', 66: 'oboe', 37: 'barrel', 34: 'rock beauty fish', 65: 'mixing bowl', 98: 'bolete', 62: 'lipstick', 89: 'split-rail fence', 32: 'black-footed ferret', 33: 'three-toed sloth', 12: 'Ibizan Hound', 20: 'Alaskan Malamute', 0: 'house finch', 84: 'tobacco shop', 54: 'fire screen', 60: 'hourglass', 4: 'harvestman', 36: 'trash can', 80: 'stage', 41: 'carousel', 93: 'trifle', 13: 'Saluki', 26: 'Arctic fox', 58: 'holster', 88: 'wok', 48: 'crate', 75: 'scoreboard', 27: 'lion', 3: 'green mamba', 70: 'photocopier', 96: 'cliff', 64: 'missile', 35: 'aircraft carrier', 24: 'Alaskan tundra wolf', 45: 'clogs', 47: 'combination lock', 51: 'dome', 23: 'Miniature Poodle', 78: 'solar thermal collector', 69: 'pencil case', 40: 'cannon', 38: 'beer bottle', 97: 'coral reef', 71: 'poncho', 44: 'bell or wind chime', 6: 'goose', 85: 'unicycle', 61: 'iPod', 77: 'snorkel', 50: 'dishcloth', 31: 'ant', 67: 'pipe organ', 17: 'Boxer', 29: 'ladybug', 16: 'Komondor', 94: 'hot dog', 28: 'meerkat', 39: 'bookstore', 83: 'tile roof', 21: 'Dalmatian', 15: 'Gordon Setter', 91: 'traffic or street sign', 76: 'slot machine', 74: 'school bus', 10: 'dugong', 92: 'consomme', 57: 'hair clip'}

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        out = []
        cache = {}
        for data_source in data_sources:
            for item in data_source:
                impath = item.impath
                if impath not in cache:
                    cache[impath] = 1
                    out.append(item)
                else:
                    if cache[impath] < num_shots:
                        cache[impath] += 1
                        out.append(item)

        return out

    def read_train_data(self, filepath):
        img_ids = {}
        out = []
        n = 0
        annotation = read_json(filepath)
        for ann in annotation:
            img_id = ann["image_id"]
            image_path = os.path.join(
                self.dataset_dir, "train2014/COCO_train2014_000000" + str(ann["image_id"]) + ".jpg"
            )
            caption = "" + pre_caption(ann["caption"], 50)
            if caption not in img_ids:
                img_ids[caption] = n
                n += 1
            item = Datum(impath=image_path, label=img_ids[caption], classname=caption)
            out.append(item)
        return out

    def read_valtest_data(self, filepath):
        out = []
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        txt_id = 0
        img_id = 0
        annotation = read_json(filepath)
        for ann in annotation:
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(pre_caption(caption, 50))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1
            img_id += 1
            image_path = os.path.join(self.dataset_dir, ann["image"])
            item = Datum(impath=image_path, label=img_id, classname=pre_caption(caption, 50))
            out.append(item)
        return out
