# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import os
import os.path as osp
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy, compute_accuracy_hf
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from torch.nn import functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


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


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "MiniImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "ITC_Flickr": "{}.",
    "ITC_Flickr5k": "{}.",
    "ITC_Mscoco": "{}.",
}
_MODELS = {
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-L/14": "openai/clip-vit-large-patch14",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = _MODELS[backbone_name]

    model = CLIPModel.from_pretrained(url)
    processor = CLIPProcessor.from_pretrained(url)
    # model.initialize_parameters()

    return model, processor


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.fc(x)
        return x


# use clip textencode
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model, processor):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.tokenizer = processor.tokenizer
        self.dtype = clip_model.dtype

    def forward(self, classname=None):
        # for small dataset, we tokenize all prompt ------- if classname is None
        # for large dataset, we tokenize (bs) prompt
        if classname is None:
            temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        else:
            temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in classname]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
        if self.cfg.TRAINER.COOP.XPU:
            prompts = prompts.to(self.cfg.TRAINER.COOP.XPU_ID)
        else:
            prompts = prompts.to("cuda")
        text_features = self.clip_model.text_model(prompts)[1]
        text_features = self.clip_model.text_projection(text_features)
        return text_features


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model, processor):
        super().__init__()
        self.visual_projection = clip_model.visual_projection
        self.image_encoder = clip_model.vision_model
        if "ITC" in cfg.DATASET.NAME:
            self.text_encoder = TextEncoder(cfg, None, clip_model, processor)
        else:
            self.text_encoder = TextEncoder(cfg, classnames, clip_model, processor)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # init adapter
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

    def forward(self, image, classname=None):
        image_features = self.image_encoder(image.type(self.dtype))[1]
        image_features = self.visual_projection(image_features)
        # apply adapter in ViT
        # x = self.adapter(image_features)

        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder(classname)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_img = logit_scale * image_features @ text_features.t()
        logits_text = logit_scale * text_features @ image_features.t()
        return logits_img, logits_text


@TRAINER_REGISTRY.register()
class Tip_Adapter(TrainerX):
    """CLIP-Adapter."""

    def build_model(self):
        self.train_flag = 0
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        if "ITC" in self.cfg.DATASET.NAME:
            classnames = None
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, processor = load_clip_to_cpu(cfg)
        clip_model.float()
        self.nceloss = nn.CrossEntropyLoss()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, processor)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # param.requires_grad_(True)
            # if 'adapter' not in name:
            #     param.requires_grad_(False)
            param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)
        # self.model.to(self.device)

        if self.cfg.TRAINER.COOP.XPU:
            torch.xpu.set_device(self.cfg.TRAINER.COOP.XPU_ID)
            self.model.xpu(self.cfg.TRAINER.COOP.XPU_ID)
            if torch.xpu.device_count() > 1:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.cfg.TRAINER.COOP.XPU_ID]
                )
        else:
            self.model.to(self.device)

        # self.optim = build_optimizer(self.model, cfg.OPTIM)
        # self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # init cache
        train_loader_cache = self.train_loader_cache
        self.cache_keys, self.cache_values = self.build_cache_model(self.cfg, self.model, train_loader_cache)

        device = self.cfg.TRAINER.COOP.XPU_ID if self.cfg.TRAINER.COOP.XPU else self.cfg.TRAINER.COOP.CUDA_ID
        self.adapter = (
            nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(clip_model.dtype).to(device)
        )

        self.adapter.weight = nn.Parameter(self.cache_keys.t())
        for name, param in self.adapter.named_parameters():
            param.requires_grad_(True)
        self.optim = build_optimizer(self.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.beta, self.alpha = cfg.TRAINER.TIP.beta, cfg.TRAINER.TIP.alpha
        self.register_model("tip_adapter", self.model, self.optim, self.sched)
        device_count = torch.cuda.device_count()
        print("=====================================")
        print(device_count)
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            torch.cuda.set_device(self.cfg.TRAINER.COOP.CUDA_ID)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.cfg.TRAINER.COOP.CUDA_ID]
            )

    def run_epoch(self):
        self.train_flag = 1
        super().run_epoch()

    def build_cache_model(self, cfg, clip_model, train_loader_cache):
        if not cfg.TRAINER.TIP.LOAD_CACHE:
            cache_keys = []
            cache_values = []
            if cfg.TRAINER.TIP.NEW:
                for i in range(0, 100):
                    cache_keys.append([])
                    cache_values.append(i)
                with torch.no_grad():
                    for augment_idx in range(cfg.TRAINER.TIP.AUGMENT_EPOCH):

                        print("Augment Epoch: {:} / {:}".format(augment_idx, cfg.TRAINER.TIP.AUGMENT_EPOCH))
                        for batch_idx, batch in enumerate(tqdm(train_loader_cache)):
                            images, label, classname = self.parse_batch_test(batch)
                            images = images.to(self.device)
                            image_features = self.get_img_embeds(images)
                            text_features = self.get_text_embeds(classname)
                            for i in range(0, image_features.shape[0]):
                                cache_keys[label[i] % 100].append(image_features[i])
                            if augment_idx == 0:
                                label = label.to(self.device)
                    for i in range(0, 100):
                        eps = 1e-6
                        stacked_vectors = torch.stack(cache_keys[i])
                        cache_keys[i] = stacked_vectors.mean(dim=0)
                cache_keys = torch.stack(cache_keys)
                cache_keys = cache_keys.permute(1, 0)
                cache_values = torch.nn.functional.one_hot(torch.tensor(cache_values).to(self.device), 100).float()
                torch.save(
                    cache_keys, cfg.TRAINER.TIP.CACHE_DIR_NEW + "/keys_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt"
                )
                torch.save(
                    cache_keys, cfg.TRAINER.TIP.CACHE_DIR + "/keys_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt"
                )
                torch.save(
                    cache_values,
                    cfg.TRAINER.TIP.CACHE_DIR_NEW + "/values_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt",
                )
                torch.save(
                    cache_values, cfg.TRAINER.TIP.CACHE_DIR + "/values_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt"
                )
                return cache_keys, cache_values

            with torch.no_grad():
                for augment_idx in range(cfg.TRAINER.TIP.AUGMENT_EPOCH):
                    train_features = []

                    print("Augment Epoch: {:} / {:}".format(augment_idx, cfg.TRAINER.TIP.AUGMENT_EPOCH))
                    for batch_idx, batch in enumerate(tqdm(train_loader_cache)):
                        images, label, classname = self.parse_batch_test(batch)
                        images = images.to(self.device)
                        image_features = self.get_img_embeds(images)

                        train_features.append(image_features)
                        if augment_idx == 0:
                            label = label.to(self.device)
                            cache_values.append(label)
                    cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)
            cache_values = torch.nn.functional.one_hot(torch.cat(cache_values, dim=0), 100).float()
            torch.save(cache_keys, cfg.TRAINER.TIP.CACHE_DIR + "/keys_" + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")
            torch.save(cache_values, cfg.TRAINER.TIP.CACHE_DIR + "/values_" + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")

        else:
            if cfg.TRAINER.TIP.NEW:
                cache_keys = []
                cache_values = []
                new_cache_keys = torch.load(
                    cfg.TRAINER.TIP.CACHE_DIR_NEW + "/keys_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt",
                    weights_only=False,
                )
                new_cache_values = torch.load(
                    cfg.TRAINER.TIP.CACHE_DIR_NEW + "/values_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt",
                    weights_only=False,
                )
                if cfg.TRAINER.TIP.NEW_DATASET:
                    new_cache_keys = new_cache_keys.permute(1, 0)
                    for i in range(0, 100):
                        cache_keys.append([])
                        cache_keys[i].append(new_cache_keys[i])
                        cache_values.append(i)
                    with torch.no_grad():
                        for augment_idx in range(cfg.TRAINER.TIP.AUGMENT_EPOCH):
                            print("Augment Epoch: {:} / {:}".format(augment_idx, cfg.TRAINER.TIP.AUGMENT_EPOCH))
                            for batch_idx, batch in enumerate(tqdm(train_loader_cache)):
                                images, label, classname = self.parse_batch_test(batch)
                                images = images.to(self.device)
                                image_features = self.get_img_embeds(images)
                                text_features = self.get_text_embeds(classname)
                                # print(image_features.shape)
                                for i in range(0, image_features.shape[0]):
                                    cache_keys[label[i] % 100].append(image_features[i])
                                if augment_idx == 0:
                                    label = label.to(self.device)
                        for i in range(0, 100):
                            eps = 1e-6
                            stacked_vectors = torch.stack(cache_keys[i])
                            cache_keys[i] = stacked_vectors.mean(dim=0)
                    cache_keys = torch.stack(cache_keys)
                    cache_keys = cache_keys.permute(1, 0)
                    cache_values = torch.nn.functional.one_hot(torch.tensor(cache_values).to(self.device), 100).float()
                    torch.save(
                        cache_keys,
                        cfg.TRAINER.TIP.CACHE_DIR_NEW + "/keys_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt",
                    )
                    torch.save(
                        cache_values,
                        cfg.TRAINER.TIP.CACHE_DIR_NEW + "/values_" + str(cfg.DATASET.NUM_SHOTS) + "shots_new.pt",
                    )
                else:
                    return new_cache_keys, new_cache_values

            else:
                cache_keys = torch.load(
                    cfg.TRAINER.TIP.CACHE_DIR + "/keys_" + str(cfg.DATASET.NUM_SHOTS) + "shots.pt", weights_only=False
                )
                cache_values = torch.load(
                    cfg.TRAINER.TIP.CACHE_DIR + "/values_" + str(cfg.DATASET.NUM_SHOTS) + "shots.pt", weights_only=False
                )

        return cache_keys, cache_values

    @torch.no_grad()
    def test(self, split=None):

        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        # split="val"
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train_u":
            data_loader = self.train_loader_u
            if "ITC" in self.cfg.DATASET.NAME:
                data_loader = self.test_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        if self.cfg.TRAINER.TIP.search_best:
            new_data_loader = self.test_loader
            images = []
            labels = []
            texts = []
            best_acc = 0.0
            beta_list = [
                i * (self.cfg.TRAINER.TIP.search_scale[0] - 0.1) / self.cfg.TRAINER.TIP.search_step[0] + 0.1
                for i in range(self.cfg.TRAINER.TIP.search_step[0])
            ]
            alpha_list = [
                i * (self.cfg.TRAINER.TIP.search_scale[1] - 0.1) / self.cfg.TRAINER.TIP.search_step[1] + 0.1
                for i in range(self.cfg.TRAINER.TIP.search_step[1])
            ]

            logit_scale = self.model.logit_scale
            for beta in beta_list:
                for alpha in alpha_list:
                    self.evaluator.reset()
                    for batch_idx, batch in enumerate(tqdm(new_data_loader)):
                        image, label, classname = self.parse_batch_test(batch)
                        image_features = self.get_img_embeds(image)
                        text_features = self.get_text_embeds(classname)

                        if self.train_flag == 1:
                            affinity = self.adapter(image_features)
                        else:
                            affinity = image_features @ self.cache_keys

                        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
                        logits = 100.0 * image_features @ text_features.t()
                        tip_logits = logits + cache_logits * alpha
                        self.evaluator.process(tip_logits, label)
                    results = self.evaluator.evaluate()
                    for k, v in results.items():
                        tag = f"{split}/{k}"
                        if "acc" in k:
                            acc = v

                    if acc > best_acc:
                        print(
                            "New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc)
                        )
                        best_acc = acc
                        self.beta = beta
                        self.alpha = alpha

        # config for ITC task
        if "ITC" in self.cfg.DATASET.NAME:
            if split == "val":
                if self.cfg.DATASET.NAME == "ITC_Flickr":
                    annotation = json.load(
                        open(os.path.join(os.path.join(self.cfg.DATASET.ROOT, "flickr"), "flickr30k_val.json"), "r")
                    )
                elif self.cfg.DATASET.NAME == "ITC_Flickr5k":
                    annotation = json.load(
                        open(os.path.join(os.path.join(self.cfg.DATASET.ROOT, "flickr5k"), "flickr5k_val.json"), "r")
                    )
                else:
                    annotation = json.load(
                        open(
                            os.path.join(os.path.join(self.cfg.DATASET.ROOT, "mscoco2014"), "coco_karpathy_val.json"),
                            "r",
                        )
                    )
            elif split == "test":
                if self.cfg.DATASET.NAME == "ITC_Flickr":
                    annotation = json.load(
                        open(os.path.join(os.path.join(self.cfg.DATASET.ROOT, "flickr"), "flickr30k_test.json"), "r")
                    )
                elif self.cfg.DATASET.NAME == "ITC_Flickr5k":
                    annotation = json.load(
                        open(os.path.join(os.path.join(self.cfg.DATASET.ROOT, "flickr5k"), "flickr5k_test.json"), "r")
                    )
                else:
                    annotation = json.load(
                        open(
                            os.path.join(os.path.join(self.cfg.DATASET.ROOT, "mscoco2014"), "coco_karpathy_test.json"),
                            "r",
                        )
                    )
            else:
                if self.cfg.DATASET.NAME == "ITC_Flickr":
                    annotation = json.load(
                        open(os.path.join(os.path.join(self.cfg.DATASET.ROOT, "flickr"), "flickr30k_test.json"), "r")
                    )
                elif self.cfg.DATASET.NAME == "ITC_Flickr5k":
                    annotation = json.load(
                        open(os.path.join(os.path.join(self.cfg.DATASET.ROOT, "flickr5k"), "flickr5k_test.json"), "r")
                    )
                else:
                    annotation = json.load(
                        open(
                            os.path.join(os.path.join(self.cfg.DATASET.ROOT, "mscoco2014"), "coco_karpathy_test.json"),
                            "r",
                        )
                    )
            results = OrderedDict()
            image = []
            text = []
            img2text = {}
            text2img = {}
            txt_id = 0
            img_id = 0
            for ann in annotation:
                img2text[img_id] = []
                for i, caption in enumerate(ann["caption"]):
                    text.append(pre_caption(caption, 50))
                    img2text[img_id].append(txt_id)
                    text2img[txt_id] = img_id
                    txt_id += 1
                img_id += 1
            text_bs = 256
            text_embeds = []
            for i in range(0, txt_id, text_bs):
                texts = text[i : min(txt_id, i + text_bs)]
                text_features = self.get_text_embeds(texts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_embeds.append(text_features)

            text_embeds = torch.cat(text_embeds, dim=0)

            image_embeds = []
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                image, label, classname = self.parse_batch_test(batch)
                logits_img, logits_text = self.model(image, classname)
                n = logits_img.shape[1]
                label = torch.from_numpy(np.arange(n)).to(self.device)
                self.evaluator.process(logits_img, label)
                img_features = self.get_img_embeds(image)
                image_embeds.append(img_features)

            image_embeds = torch.cat(image_embeds, dim=0)
            print("image_embeds", image_embeds.shape)
            print("text_embeds", text_embeds.shape)
            # self.model.logit_scale.exp() *
            score_matrix_i2t = self.model.logit_scale.exp() * image_embeds @ text_embeds.t()
            score_matrix_t2i = self.model.logit_scale.exp() * text_embeds @ image_embeds.t()

            score_matrix_i2t = score_matrix_i2t.cpu().numpy()
            score_matrix_t2i = score_matrix_t2i.cpu().numpy()
            results = self.evaluator.evaluate_flickr(score_matrix_i2t, score_matrix_t2i, text2img, img2text)
            for k, v in results.items():
                tag = f"{split}/{k}"
                if "r_mean" in k:
                    self.test_acc = v
                self.write_scalar(tag, v, self.epoch)
            print("=============================finish_test====================", flush=True)
            return list(results.values())[0]
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label, classname = self.parse_batch_test(batch)
            image_features = self.get_img_embeds(image)
            text_features = self.get_text_embeds(classname)
            logit_scale = self.model.logit_scale
            logits = image_features @ text_features.t()
            if self.train_flag == 1:
                affinity = self.adapter(image_features)
                cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
                logits_img = 100.0 * logits + cache_logits * self.alpha
            else:
                affinity = image_features @ self.cache_keys
                cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values

                logits_img = 100.0 * logits + cache_logits * self.alpha
            if "ITC" in self.cfg.DATASET.NAME:
                n = logits_img.shape[1]
                label = torch.from_numpy(np.arange(n)).to(self.device)
            self.evaluator.process(logits_img, label)
        print("best beta", self.beta, self.alpha)
        results = self.evaluator.evaluate()
        self.opt_results = results
        csv_row = []
        csv_row.append(str(self.cfg.OPTIM.LR))
        csv_row.append(str(self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE))
        if split == "test":
            csv_row.append("test")
        else:
            csv_row.append(split + str(self.epoch + 1))
        for k, v in results.items():
            tag = f"{split}/{k}"
            if "acc" in k:
                self.test_acc = v
            csv_row.append(v)
            self.write_scalar(tag, v, self.epoch)
        self.csv_data.append(csv_row)
        print(csv_row)
        if split == "test":
            filename = "./output.csv"
            with open(filename, "a+") as csvfile:
                writer = csv.writer(csvfile)
                for row in self.csv_data:
                    writer.writerow(row)
        print("=============================finish_test====================", flush=True)
        return list(results.values())[0]

    def model_inference(self, image):
        print("inference")
        self.adapter.eval()
        image_features = self.model.image_encoder(image.type(self.model.dtype))[1]
        image_features = self.model.visual_projection(image_features)
        # x = self.model.adapter(image_features)
        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.model.text_encoder(None)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = image_features @ self.text_features.t()
        if self.train_flag == 1:
            affinity = self.adapter(image_features)
            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
            tip_logits = 100.0 * logits + cache_logits * self.alpha
            return tip_logits

        affinity = image_features @ self.cache_keys
        cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values

        tip_logits = 100.0 * logits + cache_logits * self.alpha
        return tip_logits

    def forward_backward(self, batch):
        self.adapter.train()
        image, label, classname = self.parse_batch_train(batch)
        with torch.no_grad():
            image_features = self.get_img_embeds(image)
            text_features = self.get_text_embeds(classname)
        logit_scale = self.model.logit_scale
        affinity = self.adapter(image_features)
        cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
        logits_img = 100.0 / logit_scale * image_features @ text_features.t() + cache_logits * self.alpha

        if "ITC" in self.cfg.DATASET.NAME:
            n = logits_img.shape[1]
            label = torch.arange(len(logits_img)).long().to(self.device)
            loss = self.nceloss(logits_img, label)
        else:
            loss = F.cross_entropy(logits_img, label)

        self.model_backward_and_update(loss)

        if "ITC" in self.cfg.DATASET.NAME:
            loss_summary = {"loss": loss.item(), "acc": compute_accuracy_hf(logits_img, label)}
        else:
            loss_summary = {"loss": loss.item(), "acc": compute_accuracy(logits_img, label)[0].item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def get_text_embeds(self, text):
        text_features = self.model.text_encoder(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_img_embeds(self, image):
        image_features = self.model.image_encoder(image.type(self.model.dtype))[1]
        image_features = self.model.visual_projection(image_features)
        # x = self.model.adapter(image_features)

        # ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        if "ITC" in self.cfg.DATASET.NAME:
            classname = batch["classname"]
        else:
            classname = None
        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, classname

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
