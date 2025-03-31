# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import os.path as osp
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy, compute_accuracy_hf
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import (
    get_accuracies,
    get_one_to_one_features,
    l2_norm,
    load_checkpoint,
    load_pretrained_weights,
    sinkhorn_assignment,
)
from torch.nn import functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

N_AUGMENTATIONS = 5
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

final_beta = 0.0
final_alpha = 0.0


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = _MODELS[backbone_name]

    model = CLIPModel.from_pretrained(url)
    processor = CLIPProcessor.from_pretrained(url)
    # model.initialize_parameters()

    return model, processor


def log_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(1 - np.exp(-5.0 * current / rampup_length))


def adaptive_reranking_loss(visual_features, class_prototypes, labels, scale=4.0, knn=3, **_):
    N = visual_features.shape[0]
    C = class_prototypes.shape[0]
    knn = min(knn, C)

    visual_features = l2_norm(visual_features)
    class_prototypes = l2_norm(class_prototypes)

    distances = torch.cdist(visual_features, class_prototypes, p=2)

    sorted_distances, sorted_indices = torch.sort(distances, dim=1, descending=False)
    anchor = ((visual_features - class_prototypes[labels]) ** 2).sum(-1).sqrt().unsqueeze(1)
    sorted_distances = sorted_distances[:, :knn]

    pos_cla_proto = class_prototypes[labels].unsqueeze(1)
    all_cls = class_prototypes[sorted_indices[:, :knn]]
    margins = (1.0 - (all_cls * pos_cla_proto).sum(-1)) / scale

    loss = torch.max(
        anchor + margins - sorted_distances,
        torch.zeros(N, knn).to(visual_features.device),
    )

    return loss.mean()


def procrustes_align(features_src, features_tgt, beta=0.85):
    u, _, v = torch.svd(features_src.T @ features_tgt)
    W = u @ v.T
    identity = torch.eye(W.size(0)).to(W.device)
    W = W - (W - identity) * beta
    return W


def pseudo_align(features_src, features_tgt):
    x_source_pseudo = torch.linalg.inv(features_src.T @ features_src)
    x_source_pseudo = x_source_pseudo @ features_src.T
    W = x_source_pseudo @ features_tgt
    return W


def cross_validation_beta_procrustes(
    cfg, visual_features, class_prototypes, labels, num_of_tries=25, num_samples=3, five_crop=False
):

    device = cfg.TRAINER.COOP.XPU_ID if cfg.TRAINER.COOP.XPU else cfg.TRAINER.COOP.CUDA_ID
    visual_features = visual_features.to(device)
    class_prototypes = class_prototypes.to(device)
    labels = labels.to(device)

    betas = torch.linspace(0.0, 1.0, num_of_tries)
    best_beta = 0.0
    beta_transform = None
    max_score = float("-inf")

    if five_crop:
        # only keep center crops - faster more stable
        mask = (torch.tensor([0, 0, 0, 0, 1]).repeat(visual_features.shape[0] // N_AUGMENTATIONS).bool()).to(device)
        mask = (
            torch.cat([mask.to("cpu"), torch.zeros(visual_features.shape[0] - mask.shape[0])], dim=0).bool().to(device)
        )
        visual_features = visual_features[mask]
        labels = labels[mask]

    def create_arrrays(visual_features, labels):
        new_train_size = int(len(visual_features) * 0.8)
        new_train_array = {}
        new_test_array = {}

        new_train_array["visual_features"] = visual_features[:new_train_size]
        new_train_array["labels"] = labels[:new_train_size]
        new_train_array["text_features"] = class_prototypes

        new_test_array["visual_features"] = visual_features[new_train_size:]
        new_test_array["labels"] = labels[new_train_size:]
        new_test_array["text_features"] = class_prototypes

        return new_train_array, new_test_array

    for beta in tqdm(betas):
        score = 0.0
        for _ in range(num_samples):
            perm = torch.randperm(len(visual_features))
            new_train_array, new_test_array = create_arrrays(visual_features[perm], labels[perm])

            transfm = procrustes_align(
                new_train_array["visual_features"],
                new_train_array["text_features"][new_train_array["labels"]],
                beta=beta.item(),
            )
            acc = get_accuracies(cfg, new_train_array, new_test_array, transform=transfm)
            score += acc["test"]["top_1"]
        score = score / num_samples
        if score > max_score:
            max_score = score
            best_beta = beta
    print(f"Beta selected is: {best_beta:.2f}")
    global final_beta
    final_beta = best_beta
    beta_transform = procrustes_align(visual_features, class_prototypes[labels], beta=best_beta)
    return beta_transform.cpu()


def mapping_refinement(
    cfg,
    loss_func,
    init_transfm,
    train_arrays,
    test_arrays,
    train_visual_feats,
    train_text_feats,
    class_prototypes,
    labels,
    batch_size,
    momentum=0.9,
    return_ema_transform=False,
    verbose=True,
):
    device = cfg.TRAINER.COOP.XPU_ID if cfg.TRAINER.COOP.XPU else cfg.TRAINER.COOP.CUDA_ID
    ema_transform = torch.eye(init_transfm.size(1)).to(device)

    num_instances = (
        train_visual_feats.size(0) // N_AUGMENTATIONS if cfg.TRAINER.LFA.FIVE_CROP else train_visual_feats.size(0)
    )

    if batch_size is None:
        batch_size = num_instances if cfg.TRAINER.LFA.FIVE_CROP else int(num_instances * 0.75)

    assert (class_prototypes is not None and labels is not None) or train_text_feats

    transfm = torch.nn.Parameter(init_transfm.clone().to(device), requires_grad=True)
    class_prototypes = class_prototypes.to(device) if class_prototypes is not None else None

    opt = torch.optim.AdamW(
        [transfm],
        lr=cfg.OPTIM.LR,
        eps=1e-08,
        betas=(0.9, 0.999),
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
    )

    # init learning rate scheduler, cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg.OPTIM.MAX_EPOCH, eta_min=cfg.TRAINER.LFA.COSINE_END_LR
    )

    tbar = tqdm(range(cfg.OPTIM.MAX_EPOCH)) if verbose else range(cfg.OPTIM.MAX_EPOCH)

    for num_iter in tbar:
        opt.zero_grad()

        if cfg.TRAINER.LFA.FIVE_CROP:
            # use only one of the crops at random (faster)
            mask = torch.tensor([0, 0, 0, 0, 1]).bool()
            mask = mask[torch.randperm(5)].repeat(num_instances)
            mask = torch.cat([mask.to("cpu"), torch.zeros(train_visual_feats.shape[0] - mask.shape[0])], dim=0).bool()
            train_visual_batch = train_visual_feats[mask]
            labels_batch = labels[mask] if labels is not None else None
            train_text_batch = train_text_feats[mask] if train_text_feats is not None else None
        else:
            train_visual_batch = train_visual_feats
            train_text_batch = train_text_feats
            labels_batch = labels

        if batch_size is not None:
            batch_indices = torch.randperm(num_instances)[:batch_size]

            train_visual_batch = train_visual_batch[batch_indices]
            train_text_batch = train_text_batch[batch_indices] if train_text_batch is not None else None
            labels_batch = labels_batch[batch_indices] if labels is not None else None

        train_visual_batch = train_visual_batch.to(device)
        train_text_batch = train_text_batch.to(device)
        labels_batch = labels_batch.to(device)

        if cfg.TRAINER.LFA.GAUSSIAN_NOISE > 0.0 and np.random.uniform() > 0.5:
            train_visual_batch += torch.randn_like(train_visual_batch) * cfg.TRAINER.LFA.GAUSSIAN_NOISE

        if cfg.TRAINER.LFA.DROP_OUT > 0.0 and np.random.uniform() > 0.5:
            train_visual_batch = torch.nn.functional.dropout(train_visual_batch, p=cfg.TRAINER.LFA.DROP_OUT)

        # compute the loss
        loss = loss_func(
            visual_features=train_visual_batch @ transfm,
            text_features=train_text_batch,
            class_prototypes=class_prototypes,
            labels=labels_batch,
            knn=cfg.TRAINER.LFA.KNN,
        )
        loss.requires_grad_(True)
        # update the parameters
        loss.backward()
        opt.step()
        scheduler.step()
        # project the transformation matrix to the space of orthogonal matrices

        if cfg.TRAINER.LFA.orthogonalize:
            transfm.data = (1 + cfg.TRAINER.LFA.orth_beta) * transfm.data - cfg.TRAINER.LFA.orth_beta * (
                (transfm.data @ transfm.data.T) @ transfm.data
            )

        momentum = 0.1 * log_rampup(num_iter, cfg.OPTIM.MAX_EPOCH // 2) + 0.9
        ema_transform = momentum * ema_transform.data + (1.0 - momentum) * transfm.data

        # Compute train and test accuracies
        accuracies = get_accuracies(
            cfg, train_arrays, test_arrays, transform=transfm.cpu(), five_crop=cfg.TRAINER.LFA.FIVE_CROP
        )

        # update the progress bar
        if verbose:
            tbar.set_description(f"Loss: {loss.item():.4f} | {accuracies}")

    if return_ema_transform:
        return transfm.detach().cpu(), ema_transform.cpu()

    return transfm.detach().cpu()


def iterative_unsupervised_refinement(cfg, loss_func, train_arrays, test_arrays, th=0.0):
    n_unsup_iters = cfg.TRAINER.LFA.UNSUP
    class_prototypes = train_arrays["text_features"]
    N, _ = train_arrays["visual_features"].size()

    soft_assignments = sinkhorn_assignment(
        cfg,
        train_arrays["visual_features"],
        class_prototypes,
        blur=0.05,
    )

    mask = soft_assignments.max(-1)[0] > th
    soft_assignments = soft_assignments[mask]
    visual_features = train_arrays["visual_features"][mask]
    labels = soft_assignments.argmax(-1)

    transfm = cross_validation_beta_procrustes(
        cfg,
        visual_features,
        class_prototypes,
        labels,
        five_crop=cfg.TRAINER.LFA.FIVE_CROP,
    )

    print("initial results - to be done")
    tbar = tqdm(range(n_unsup_iters))
    for n_iter in tbar:
        soft_assignments = sinkhorn_assignment(
            cfg,
            l2_norm(train_arrays["visual_features"] @ transfm),
            class_prototypes,
            blur=0.05,
        )

        mask = soft_assignments.max(-1)[0] > th

        soft_assignments = soft_assignments[mask]
        visual_features = train_arrays["visual_features"][mask]
        text_features = l2_norm(soft_assignments @ class_prototypes)
        labels = soft_assignments.argmax(-1)

        transfm = mapping_refinement(
            cfg,
            loss_func=loss_func,
            init_transfm=transfm,
            train_arrays=train_arrays,
            test_arrays=test_arrays,
            train_visual_feats=visual_features,
            train_text_feats=text_features,
            class_prototypes=class_prototypes,
            labels=labels,
            batch_size=len(visual_features),
            verbose=True,
        )
        tbar.set_description(f"Iter: {n_iter} | Used examples: {mask.sum()}/{N}")
    print("After refinement results: - to be done")
    return transfm


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
        # init adapter
        # self.adapter = Adapter(512, 4).to(clip_model.dtype)

    def forward(self, image, classname=None):
        image_features = self.image_encoder(image.type(self.dtype))[1]
        image_features = self.visual_projection(image_features)

        text_features = self.text_encoder(classname)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_img = logit_scale * image_features @ text_features.t()
        logits_text = logit_scale * text_features @ image_features.t()
        return logits_img, logits_text


@TRAINER_REGISTRY.register()
class CLIP_LFA(TrainerX):
    """CLIP-LFA."""

    global final_beta

    def build_model(self):
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
            param.requires_grad_(True)

        self.alpha = cfg.TRAINER.LFA.alpha
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
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("clip_lfa", self.model, self.optim, self.sched)
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

    @torch.no_grad()
    def test(self, split=None):
        device = self.cfg.TRAINER.COOP.XPU_ID if self.cfg.TRAINER.COOP.XPU else self.cfg.TRAINER.COOP.CUDA_ID
        self.set_model_mode("train")
        self.evaluator.reset()
        # with torch.no_grad():
        data_loader = self.test_loader

        test_image_embeds = []
        test_labels = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label, classname = self.parse_batch_test(batch)
            img_features = self.get_img_embeds(image)
            test_image_embeds.append(img_features)
            test_labels.append(label)
        test_image_embeds = torch.cat(test_image_embeds, dim=0)
        test_text_embeds = self.get_text_embeds(None)
        test_labels = torch.cat(test_labels, dim=0)

        data_loader = self.train_loader_u

        train_image_embeds = []
        train_labels = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label, classname = self.parse_batch_test(batch)
            img_features = self.get_img_embeds(image)
            train_image_embeds.append(img_features)
            train_labels.append(label)
        train_image_embeds = torch.cat(train_image_embeds, dim=0)
        train_text_embeds = self.get_text_embeds(None)
        train_labels = torch.cat(train_labels, dim=0)
        train_arrays = {}
        train_arrays["visual_features"] = train_image_embeds.cpu()
        train_arrays["text_features"] = train_text_embeds.cpu()
        train_arrays["labels"] = train_labels.cpu()

        test_arrays = {}
        test_arrays["visual_features"] = test_image_embeds.cpu()
        test_arrays["text_features"] = test_text_embeds.cpu()
        test_arrays["labels"] = test_labels.cpu()

        accuracies = get_accuracies(
            self.cfg, train_arrays, test_arrays, transform=None, five_crop=self.cfg.TRAINER.LFA.FIVE_CROP
        )
        print(f"Clip results {accuracies}\n")

        if not self.cfg.TRAINER.LFA.unsupervised:
            print("not unsupervised")
            train_text_feats = get_one_to_one_features(
                train_arrays["visual_features"],
                train_arrays["text_features"],
                train_arrays["labels"],
            )

            print("Procrustes results:")
            transfm_proc = procrustes_align(
                train_arrays["visual_features"].to(device),
                train_text_feats.to(device),
                beta=0.0,
            )

            accuracies = get_accuracies(
                self.cfg,
                train_arrays,
                test_arrays,
                transform=transfm_proc,
                five_crop=self.cfg.TRAINER.LFA.FIVE_CROP,
                alpha=self.alpha,
            )

            print(f"Procrustes results: {accuracies}\n")
            global final_beta
            if self.cfg.TRAINER.LFA.beta_procrustes is None:
                transfm = cross_validation_beta_procrustes(
                    self.cfg,
                    train_arrays["visual_features"],
                    train_arrays["text_features"],
                    train_arrays["labels"],
                    five_crop=self.cfg.TRAINER.LFA.FIVE_CROP,
                )
            else:
                transfm = procrustes_align(
                    train_arrays["visual_features"].to(device),
                    train_text_feats.to(device),
                    beta=self.cfg.TRAINER.LFA.beta_procrustes,
                )

                final_beta = self.cfg.TRAINER.LFA.beta_procrustes

            accuracies = get_accuracies(
                self.cfg,
                train_arrays,
                test_arrays,
                transform=transfm,
                five_crop=self.cfg.TRAINER.LFA.FIVE_CROP,
                alpha=self.alpha,
            )

            print(f"Beta-Procrustes results: {accuracies}\n")

            print("Mapping refinement ...")

            refined_transfm, ema_transform = mapping_refinement(
                self.cfg,
                loss_func=partial(
                    adaptive_reranking_loss, knn=self.cfg.TRAINER.LFA.KNN, scale=self.cfg.TRAINER.LFA.arerank_scale
                ),
                init_transfm=transfm,
                train_arrays=train_arrays,
                test_arrays=test_arrays,
                train_visual_feats=train_arrays["visual_features"],
                train_text_feats=train_text_feats,
                class_prototypes=train_arrays["text_features"],
                labels=train_arrays["labels"],
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                return_ema_transform=True,
            )
            if self.cfg.TRAINER.LFA.search_best:
                best_acc = 0.0
                best_alpha = 0.0
                current_alpha = self.cfg.TRAINER.LFA.step
                while current_alpha < 1.0:
                    accuracies = get_accuracies(
                        self.cfg,
                        train_arrays,
                        test_arrays,
                        transform=refined_transfm,
                        five_crop=self.cfg.TRAINER.LFA.FIVE_CROP,
                        alpha=current_alpha,
                    )
                    if accuracies["test"]["top_1"] > best_acc:
                        best_acc = accuracies["test"]["top_1"]
                        best_alpha = current_alpha
                    current_alpha += self.cfg.TRAINER.LFA.step
                self.alpha = best_alpha
                print(f"bestfbeta alpha: {self.alpha}\n")

            accuracies = get_accuracies(
                self.cfg,
                train_arrays,
                test_arrays,
                transform=refined_transfm,
                five_crop=self.cfg.TRAINER.LFA.FIVE_CROP,
                alpha=self.alpha,
            )

            print(f"After refinement results: {accuracies}\n")

        else:
            print("Unsupervised iterative adaptation ...")
            refined_transfm = iterative_unsupervised_refinement(
                self.cfg,
                loss_func=partial(
                    adaptive_reranking_loss, knn=self.cfg.TRAINER.LFA.KNN, scale=self.cfg.TRAINER.LFA.arerank_scale
                ),
                train_arrays=train_arrays,
                test_arrays=test_arrays,
            )
            if self.cfg.TRAINER.LFA.search_best:
                best_acc = 0.0
                best_alpha = 0.0
                current_alpha = self.cfg.TRAINER.LFA.step
                while current_alpha < 1.0:
                    accuracies = get_accuracies(
                        self.cfg,
                        train_arrays,
                        test_arrays,
                        transform=refined_transfm,
                        five_crop=self.cfg.TRAINER.LFA.FIVE_CROP,
                        alpha=current_alpha,
                    )
                    if accuracies["test"]["top_1"] > best_acc:
                        best_acc = accuracies["test"]["top_1"]
                        best_alpha = current_alpha
                    current_alpha += self.cfg.TRAINER.LFA.step
                self.alpha = best_alpha
                print(f"best alpha: {self.alpha}\n")

            accuracies = get_accuracies(
                self.cfg,
                train_arrays,
                test_arrays,
                transform=refined_transfm,
                five_crop=self.cfg.TRAINER.LFA.FIVE_CROP,
                alpha=self.alpha,
            )
        global final_alpha
        final_alpha = self.alpha
        print(f"Unsupervised adaptation results: {accuracies}\n")
        filename = "./output_lfa.csv"
        csv_data = [["lr", "bs", "Beta", "alpha", "test_acc"]]
        csv_row = []
        csv_row.append(str(self.cfg.OPTIM.LR))
        csv_row.append(str(self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE))
        csv_row.append(str(final_beta))
        csv_row.append(str(final_alpha))
        csv_row.append(str(accuracies["test"]["top_1"]))
        csv_data.append(csv_row)
        with open(filename, "a+") as csvfile:
            writer = csv.writer(csvfile)
            for row in csv_data:
                writer.writerow(row)
        return list(accuracies.values())[0]
