# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os.path as osp
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from torch.nn import Dropout
from torch.nn import functional as F

try:
    from clip import clip
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
except ImportError:
    from ..clip import clip
    from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


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
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    model = clip.build_model(state_dict or model.state_dict())
    # model.initialize_parameters()

    return model


class PromptTransformer(nn.Module):
    def __init__(self, cfg, classnames, transformer_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        # get transformers info from clip
        self.width = transformer_model.width
        self.layers = transformer_model.layers
        self.resblocks = transformer_model.resblocks
        # get prompt info from cfg
        cfg_imsize = cfg.INPUT.SIZE[0]
        patch_size = (16, 16)
        self.prompt_length = cfg.TRAINER.COOP.N_PLN
        print("================VPT config==========")
        print(f"prompt_length *{self.prompt_length}*")
        print(f"use Deep prompt fine tune? *{self.cfg.TRAINER.COOP.PMT_DEEP}*")
        self.promptt_dropout = Dropout(cfg.TRAINER.COOP.N_PDT)

        prompt_dim = 384
        hidden_size = 768
        self.promptt_proj = nn.Linear(prompt_dim, hidden_size)
        val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # init prompt embedding for first layer
        self.promptt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, prompt_dim))
        nn.init.uniform_(self.promptt_embeddings.data, -val, val)
        if self.cfg.TRAINER.COOP.PMT_DEEP:
            self.total_d_layer = transformer_model.layers - 1
            # init prompt embedding for VPT Deep
            self.deep_promptt_embeddings = nn.Parameter(torch.zeros(self.total_d_layer, self.prompt_length, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_promptt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        B = x.shape[0]
        # add prompt embedding, (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        x = torch.cat(
            (
                x[:, :1, :],
                self.promptt_dropout(self.promptt_proj(self.promptt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :],
            ),
            dim=1,
        )
        # print("================after incorporate================")
        # print(x.shape)
        return x

    def forward_deep(self, x: torch.Tensor):
        attn_weights = []
        hidden_states = None
        weights = None
        B = x.shape[1]
        # print("=====================")
        # print(self.deep_promptt_embeddings)
        for i in range(self.total_d_layer + 1):
            # print(i)
            if i == 0:
                x = self.resblocks[i](x)
                # print(x.shape)
            else:
                if i <= self.deep_promptt_embeddings.shape[0]:
                    deep_prompt_emb = self.promptt_dropout(
                        self.promptt_proj(self.deep_promptt_embeddings[i - 1]).expand(B, -1, -1)
                    )
                    deep_prompt_emb = deep_prompt_emb.permute(1, 0, 2)
                    # update prompt embedding, (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
                    x = torch.cat((x[:1, :, :], deep_prompt_emb, x[(1 + self.prompt_length) :, :, :]), dim=0)
                x = self.resblocks[i](x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.incorporate_prompt(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.cfg.TRAINER.COOP.PMT_DEEP:
            return self.forward_deep(x)
        else:
            return self.resblocks(x)


class PromptedVisionTransformer(nn.Module):
    def __init__(self, cfg, classnames, transformer_model, dtype):
        super(PromptedVisionTransformer, self).__init__()
        self.cfg = cfg
        self.transformer_model = transformer_model
        self.dtype = dtype
        self.input_resolution = transformer_model.input_resolution
        self.output_dim = transformer_model.output_dim
        self.conv1 = transformer_model.conv1

        # scale = width ** -0.5
        self.class_embedding = transformer_model.class_embedding
        self.positional_embedding = transformer_model.positional_embedding
        self.ln_pre = transformer_model.ln_pre

        self.transformer = PromptTransformer(cfg, classnames, transformer_model.transformer)
        self.ln_post = transformer_model.ln_post
        self.proj = transformer_model.proj

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        print(x.device)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        if self.cfg.TRAINER.COOP.XPU:
            prompts = prompts.to(self.cfg.TRAINER.COOP.XPU_ID)
        else:
            prompts = prompts.to("cuda")
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = PromptedVisionTransformer(cfg, classnames, clip_model.visual, clip_model.dtype)
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # self.adapter = Adapter(512, 4).to(clip_model.dtype)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIP_VPT(TrainerX):
    """CLIP-VPT."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning on gradients for VPT")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "promptt" in name:
                param.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        if self.cfg.TRAINER.COOP.XPU:
            torch.xpu.set_device(self.cfg.TRAINER.COOP.XPU_ID)
            self.model.xpu(self.cfg.TRAINER.COOP.XPU_ID)
            if torch.xpu.device_count() > 1:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.cfg.TRAINER.COOP.XPU_ID]
                )
        else:
            self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("clip_vpt", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        print("=====================================")
        print(device_count)
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            torch.cuda.set_device(self.cfg.TRAINER.COOP.CUDA_ID)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.cfg.TRAINER.COOP.CUDA_ID]
            )

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        # print(label.shape)
        # print(label)
        output = self.model(image)
        # print(output.shape)
        # print(output)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {"loss": loss.item(), "acc": compute_accuracy(output, label)[0].item()}
        # print(loss_summary)
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
