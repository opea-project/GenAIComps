# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os.path as osp
from functools import reduce
from operator import mul
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy, compute_accuracy_hf
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from torch.nn import Dropout
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

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


class PromptEncoder(nn.Module):
    def __init__(self, cfg, classnames, transformer_model):
        super().__init__()
        self.config = transformer_model.config
        self.layers = transformer_model.layers
        self.gradient_checkpointing = transformer_model.gradient_checkpointing
        self.cfg = cfg
        self.classnames = classnames
        # get transformers info from clip
        # get prompt info from cfg
        cfg_imsize = cfg.INPUT.SIZE[0]
        patch_size = (16, 16)
        self.prompt_length = cfg.TRAINER.COOP.N_PLN
        print("================VPT config==========")
        print(f"prompt_length *{self.prompt_length}*")
        print(f"use Deep prompt fine tune? *{self.cfg.TRAINER.COOP.PMT_DEEP}*")
        self.promptt_dropout = Dropout(cfg.TRAINER.COOP.N_PDT)

        hidden_size = transformer_model.config.hidden_size
        prompt_dim = int(hidden_size / 2)
        self.promptt_proj = nn.Linear(prompt_dim, hidden_size)
        val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # init prompt embedding for first layer
        self.promptt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, prompt_dim))
        nn.init.uniform_(self.promptt_embeddings.data, -val, val)
        if self.cfg.TRAINER.COOP.PMT_DEEP:
            self.total_d_layer = self.config.num_hidden_layers - 1
            # init prompt embedding for VPT Deep
            self.deep_promptt_embeddings = nn.Parameter(torch.zeros(self.total_d_layer, self.prompt_length, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_promptt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # print("================before incorporate================")
        # print(x.shape)
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

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # add prompt in layer
        hidden_states = inputs_embeds
        hidden_states = self.incorporate_prompt(hidden_states)
        for idx, encoder_layer in enumerate(self.layers):

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                B = hidden_states.shape[0]
                # add prompt in first layer
                # or
                # add prompt in all layer if cfg.TRAINER.COOP.PMT_DEEP
                if idx == 0 or (not self.cfg.TRAINER.COOP.PMT_DEEP):
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        output_attentions=output_attentions,
                    )
                else:
                    if idx <= self.deep_promptt_embeddings.shape[0]:
                        deep_prompt_emb = self.promptt_dropout(
                            self.promptt_proj(self.deep_promptt_embeddings[idx - 1]).expand(B, -1, -1)
                        )

                        hidden_states = torch.cat(
                            (hidden_states[:, :1, :], deep_prompt_emb, hidden_states[:, (1 + self.prompt_length) :, :]),
                            dim=1,
                        )
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        causal_attention_mask,
                        output_attentions=output_attentions,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class PromptedVisionTransformer(nn.Module):
    def __init__(self, cfg, classnames, transformer_model):
        super(PromptedVisionTransformer, self).__init__()
        self.config = transformer_model.config
        embed_dim = self.config.hidden_size
        self.embeddings = transformer_model.embeddings
        self.pre_layrnorm = transformer_model.pre_layrnorm
        self.post_layernorm = transformer_model.post_layernorm
        self.cfg = cfg
        self.transformer_model = transformer_model
        self.encoder = PromptEncoder(cfg, classnames, transformer_model.encoder)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PromptedVisionModel(nn.Module):
    def __init__(self, cfg, classnames, transformer_model, dtype):
        super(PromptedVisionModel, self).__init__()
        self.config = transformer_model.config
        self.cfg = cfg
        self.transformer_model = transformer_model
        self.dtype = dtype
        # self.vision_model = transformer_model.vision_model
        self.vision_model = PromptedVisionTransformer(cfg, classnames, transformer_model.vision_model)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model, processor):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.tokenizer = processor.tokenizer
        self.dtype = clip_model.dtype

    def forward(self, classname=None):
        if classname is None:
            temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        else:
            temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
            prompts = [temp.format(c.replace("_", " ")) for c in classname]
        prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
        # print(self.cfg.TRAINER.COOP.XPU)
        if self.cfg.TRAINER.COOP.XPU:
            prompts = prompts.to(self.cfg.TRAINER.COOP.XPU_ID)
        else:
            prompts = prompts.to(self.cfg.TRAINER.COOP.CUDA_ID)
        text_features = self.clip_model.text_model(prompts)[1]
        text_features = self.clip_model.text_projection(text_features)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model, processor):
        super().__init__()
        self.visual_projection = clip_model.visual_projection
        self.image_encoder = PromptedVisionModel(cfg, classnames, clip_model, clip_model.dtype)
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

        text_features = self.text_encoder(classname)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_img = logit_scale * image_features @ text_features.t()
        logits_text = logits_img.t()
        return logits_img, logits_text


@TRAINER_REGISTRY.register()
class CLIP_VPT_hf(TrainerX):
    """CLIP-VPT_hf."""

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, processor = load_clip_to_cpu(cfg)
        clip_model.float()
        self.nceloss = nn.CrossEntropyLoss()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, processor)

        print("Turning on gradients for VPT")
        for name, param in self.model.named_parameters():
            # print(name)
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

        self.register_model("clip_vpt_hf", self.model, self.optim, self.sched)

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
        image, label, classname = self.parse_batch_train(batch)

        logits_img, logits_text = self.model(image, classname)
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
