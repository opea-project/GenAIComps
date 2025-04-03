# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import os.path as osp

import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy, compute_accuracy_hf
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

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
            prompts = prompts.to(self.cfg.TRAINER.COOP.CUDA_ID)
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
        # self.adapter = Adapter(512, 4).to(clip_model.dtype)

    def forward(self, image, classname=None):
        image_features = self.image_encoder(image.type(self.dtype))[1]
        image_features = self.visual_projection(image_features)
        x = image_features

        text_features = self.text_encoder(classname)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_img = logit_scale * image_features @ text_features.t()
        logits_text = logits_img.t()
        return logits_img, logits_text


@TRAINER_REGISTRY.register()
class CLIP_Bias_hf(TrainerX):
    """CLIP-Bias-hf."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, processor = load_clip_to_cpu(cfg)
        clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, processor)
        self.nceloss = nn.CrossEntropyLoss()
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # if 'adapter' not in name:
            param.requires_grad_(False)

        print("Turning on gradients in bias layer")
        print(self.cfg.BIAS.BIAS_TERMS)
        print(self.cfg.BIAS.BIAS_TERMS_EXCLUDE)
        if len(self.cfg.BIAS.BIAS_TERMS) == 0:
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    param.requires_grad_(True)
        else:
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    for str in self.cfg.BIAS.BIAS_TERMS:
                        if str in name:
                            param.requires_grad_(True)

        if len(self.cfg.BIAS.BIAS_TERMS_EXCLUDE) != 0:
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    for str in self.cfg.BIAS.BIAS_TERMS_EXCLUDE:
                        if str in name:
                            param.requires_grad_(False)
        # code for A770
        if self.cfg.TRAINER.COOP.XPU:
            torch.xpu.set_device(self.cfg.TRAINER.COOP.XPU_ID)
            self.model.xpu(self.cfg.TRAINER.COOP.XPU_ID)
            if torch.xpu.device_count() > 1:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.cfg.TRAINER.COOP.XPU_ID]
                )
        else:
            self.model.to(self.device)

        # keey the origin angle for pre-train model
        if cfg.MODEL.ABS:
            self.vec_ori = {}
            self.ABS_DICT = set()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    name = name.replace(".weight", "")
                    name = name.replace(".bias", "")

                    self.ABS_DICT.add(name)
                    if self.vec_ori.get(name) is None:
                        self.vec_ori[name] = copy.deepcopy(param.data.reshape(-1))
                    else:
                        self.vec_ori[name] = copy.deepcopy(torch.cat([self.vec_ori[name], param.data.reshape(-1)]))

        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        if self.cfg.TRAINER.COOP.XPU:
            import intel_extension_for_pytorch as ipex

            self.model, self.optim = ipex.optimize(model=self.model, optimizer=self.optim, level="O1")
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # print(self.sched)

        self.register_model("clip_bias_open", self.model, self.optim, self.sched)
        # self.set_model_mode("train")
        # for name, param in self.model.named_parameters():
        #     print(name,param.requires_grad)
        device_count = torch.cuda.device_count()
        print("=====================================")
        print(device_count)
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            torch.cuda.set_device(self.cfg.TRAINER.COOP.CUDA_ID)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.cfg.TRAINER.COOP.CUDA_ID]
            )

    def get_text_embeds(self, text):
        text_features = self.model.text_encoder(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_img_embeds(self, image):
        image_features = self.model.image_encoder(image.type(self.model.dtype))[1]
        image_features = self.model.visual_projection(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward_backward(self, batch):
        image, label, classname = self.parse_batch_train(batch)
        logits_img, logits_text = self.model(image, classname)
        if "ITC" in self.cfg.DATASET.NAME:
            n = logits_img.shape[1]
            label = torch.arange(len(logits_img)).long().to(self.device)
            loss = (self.nceloss(logits_img, label) + self.nceloss(logits_text, label)) / 2.0
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
