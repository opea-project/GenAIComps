# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import csv
import datetime
import gc
import json
import os
import os.path as osp
import re
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dassl.data import DataManager
from dassl.evaluation import build_evaluator
from dassl.modeling import build_backbone, build_head
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import (
    AverageMeter,
    MetricMeter,
    count_num_param,
    load_checkpoint,
    load_pretrained_weights,
    mkdir_if_missing,
    resume_from_checkpoint,
    save_checkpoint,
    tolist_if_not,
)


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


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification."""

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self.test_acc = 0.0
        self.test_time_epoch = 100000.0

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError("Cannot assign model before super().__init__() call")

        if self.__dict__.get("_optims") is None:
            raise AttributeError("Cannot assign optim before super().__init__() call")

        if self.__dict__.get("_scheds") is None:
            raise AttributeError("Cannot assign sched before super().__init__() call")

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            if self.cfg.TRAINER.COOP.XPU:
                start_epoch = resume_from_checkpoint(
                    path, self._models[name], self._optims[name], self._scheds[name], self.cfg.TRAINER.COOP.XPU_ID
                )
            else:
                start_epoch = resume_from_checkpoint(
                    path, self._models[name], self._optims[name], self._scheds[name], self.cfg.TRAINER.COOP.CUDA_ID
                )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})")
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        print("==============start close_writer")
        if self._writer is not None:
            self._writer.close()
        print("==============close_writer")

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        if (
            self.cfg.TRAINER.COOP.XPU and self.cfg.TRAINER.COOP.XPU_ID == "xpu:0"
        ) or self.cfg.TRAINER.COOP.CUDA_ID == "cuda:0":
            self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                if torch.isnan(param.data).sum() > 0:
                    print("hit nan after step()")
                    param.data = torch.where(torch.isnan(param.data), torch.full_like(param.data, 0.0), param.data)


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device(cfg.TRAINER.COOP.CUDA_ID)
        elif cfg.TRAINER.COOP.XPU:
            # import intel_extension_for_pytorch as ipex
            self.device = torch.device(cfg.TRAINER.COOP.XPU_ID)
            # print("self.device", self.device)
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf
        self.txt2img = {}
        self.img2txt = {}
        self.csv_data = [["bs", "lr", "method", "accuracy", "error_rate", "macro_f1"]]

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.train_loader_cache = dm.train_loader_cache
        self.train_loader_cache_f = dm.train_loader_cache_f

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        # print(111)
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST

        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            # print(curr_result)
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch, self.output_dir, val_result=curr_result, model_name="model-best.pth.tar")
        if (
            self.cfg.TRAINER.COOP.XPU and self.cfg.TRAINER.COOP.XPU_ID == "xpu:0"
        ) or self.cfg.TRAINER.COOP.CUDA_ID == "cuda:0":
            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

    def get_text_embeds(self, text):
        raise NotImplementedError

    def get_img_embeds(self, image):
        raise NotImplementedError

    # test code for CLIP
    @torch.no_grad()
    def test(self, split=None):
        print("test")
        """A generic testing pipeline."""
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
            logits_img, logits_text = self.model(image, classname)
            if "ITC" in self.cfg.DATASET.NAME:
                n = logits_img.shape[1]
                label = torch.from_numpy(np.arange(n)).to(self.device)
            self.evaluator.process(logits_img, label)

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
        if split == "test":
            filename = "./output.csv"
            with open(filename, "a+") as csvfile:
                writer = csv.writer(csvfile)
                for row in self.csv_data:
                    writer.writerow(row)
        print("=============================finish_test====================", flush=True)
        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        if "ITC" in self.cfg.DATASET.NAME:
            classname = batch["classname"]
        else:
            classname = None
        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, classname

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):

        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info), flush=True)

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        # if (self.epoch + 1) % 1 ==0:
        #     self.test()
        #     self.test(split="val")
        #     self.set_model_mode("train")

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    # run epoch code used in CLIP
    def run_epoch(self):
        # ABS search
        if self.cfg.MODEL.ABS and self.epoch == self.cfg.OPTIM.WARMUP_EPOCH:
            print("=====calculate Angle=====")
            cosine = torch.nn.CosineSimilarity(dim=0)
            vec_new = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if vec_new.get(name) is None:
                        vec_new[name] = param.data.reshape(-1)
                    else:
                        vec_new[name] = torch.cat([vec_new[name], param.data.reshape(-1)])
            angle = {}
            listt = []
            # keep number of train layer for each group
            if self.cfg.MODEL.ABS_GROUP == True:
                for gname in self.cfg.MODEL.ABS_GROUP_NAME:
                    angle[gname] = {}
                angle["else"] = {}
                for name in self.ABS_DICT:
                    flag = 0
                    x = torch.acos(cosine(self.vec_ori[name], vec_new[name]))
                    if torch.isnan(x):
                        x = torch.where(torch.isnan(x), torch.full_like(x, -1.0), x)
                    for gname in self.cfg.MODEL.ABS_GROUP_NAME:
                        if gname in name:
                            flag = 1
                            angle[gname][x.to("cpu")] = name
                    if flag == 0:
                        angle["else"][x.to("cpu")] = name
                for gname in angle.keys():
                    Oangle = OrderedDict(sorted(angle[gname].items(), reverse=self.cfg.MODEL.ABS_TOP))
                    drop_angle = dict(list(Oangle.items())[self.cfg.MODEL.ABS_KEEP :])
                    print(f"=====Keep {self.cfg.MODEL.ABS_KEEP} train layer for {gname} group=====")
                    for name, param in self.model.named_parameters():
                        if param.requires_grad == True:
                            if name in Oangle.values():
                                if name in drop_angle.values():
                                    param.requires_grad = False
                                    param.data = copy.deepcopy(self.vec_ori[name].reshape(param.data.shape))
                                    param.grad = None
                                    del param.grad
                                else:
                                    print(name)
                del Oangle
                del drop_angle
            # keep number of train layer
            else:
                for name in self.ABS_DICT:
                    x = torch.acos(cosine(self.vec_ori[name], vec_new[name]))
                    if torch.isnan(x):
                        x = torch.where(torch.isnan(x), torch.full_like(x, -1.0), x)
                    angle[x.to("cpu")] = name

                angle = OrderedDict(sorted(angle.items(), reverse=self.cfg.MODEL.ABS_TOP))
                drop_angle = dict(list(angle.items())[self.cfg.MODEL.ABS_KEEP :])
                print(f"=====Keep {self.cfg.MODEL.ABS_KEEP} train layer=====")
                for name, param in self.model.named_parameters():
                    if param.requires_grad == True:
                        if name in drop_angle.values():
                            param.requires_grad = False
                            del param.grad
                            param.data.to("cpu")
                            param.data = copy.deepcopy(self.vec_ori[name].reshape(param.data.shape))
                        else:
                            print(name)
                del drop_angle

            self.model.to("cpu")
            del self.vec_ori
            del self.optim
            del self.sched
            del angle
            del listt
            del vec_new
            if torch.cuda.is_available() and self.cfg.USE_CUDA:
                torch.cuda.empty_cache()
            elif self.cfg.TRAINER.COOP.XPU:
                torch.xpu.empty_cache()
            gc.collect()
            self.model.to(self.device)
            self.optim = build_optimizer(self.model, self.cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)

        if self.cfg.TRAINER.COOP.XPU and torch.xpu.device_count() > 1:
            self.train_loader_x.sampler.set_epoch(self.epoch)

        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # print(self.train_loader_x.shape)
        self.num_batches = len(self.train_loader_x)
        if self.cfg.TRAINER.COOP.Max_Batch != 0:
            print(f"Max train batch *{self.cfg.TRAINER.COOP.Max_Batch}* set")
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            if self.cfg.TRAINER.COOP.Max_Batch != 0 and self.cfg.TRAINER.COOP.Max_Batch < self.batch_idx:
                continue
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.test_time_epoch = self.num_batches * batch_time.avg
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info), flush=True)

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        if self.cfg.TRAINER.COOP.ACC != 0 and (self.epoch + 1) % self.cfg.TRAINER.COOP.ACC == 0:
            self.test(split="val")
            self.test(split="train_u")

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
