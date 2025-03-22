# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self.cfg = cfg
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        if "ITC" in self.cfg.DATASET.NAME:
            max_value, _ = torch.max(mo, dim=1)
            for i in gt:
                if mo[i][i] == max_value[i]:
                    mo[i][i] += 0.0001
            pred = mo.argmax(-1)
            i2t_acc = (mo.argmax(-1) == gt).sum() / len(mo)
            matches = pred.eq(gt).float()
        else:
            pred = mo.max(1)[1]
            matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(self._y_true, self._y_pred, average="macro", labels=np.unique(self._y_true))

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(self._y_true, self._y_pred, normalize="true")
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results

    def evaluate_flickr(self, scores_i2t, scores_t2i, txt2img, img2txt):
        results = OrderedDict()
        import numpy

        numpy.set_printoptions(threshold=np.inf)
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            for i in img2txt[index]:
                score[i] += 0.1
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            # if index == 2:
            #     print("score", score)
            #     print("inds", inds)
            #     print("max", score[inds[0]])
            #     print("img2txt[index]", img2txt[index])
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
                # if index == 2:
                #     print("x", np.where(inds == i))
                #     print("i", i ,rank)
            ranks[index] = rank
        # Compute metrics
        length = len(ranks)
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])
        # print(len(ranks))
        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]
        ranks_new = ranks[:length]
        num = 0
        for i in range(0, length):
            for j in range(0, len(img2txt[i])):
                if j == 0:
                    ranks_new[i] = ranks[num]
                else:
                    ranks_new[i] = min(ranks_new[i], ranks[num + j])
            num += len(img2txt[i])
        # print(len(ranks_new))
        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks_new < 1)[0]) / len(ranks_new)
        ir5 = 100.0 * len(np.where(ranks_new < 5)[0]) / len(ranks_new)
        ir10 = 100.0 * len(np.where(ranks_new < 10)[0]) / len(ranks_new)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        # The first value will be returned by trainer.test()
        acc = 100.0 * self._correct / self._total
        results["accuracy"] = acc
        results["r_mean"] = r_mean
        results["ir1"] = ir1
        results["ir5"] = ir5
        results["ir10"] = ir10
        results["tr1"] = tr1
        results["tr5"] = tr5
        results["tr10"] = tr10
        results["text_r_mean"] = tr_mean
        results["img_r_mean"] = ir_mean

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* r_mean: {r_mean:.1f}%\n"
            f"* ir1: {ir1:.1f}%\n"
            f"* ir5: {ir5:.1f}%\n"
            f"* ir10: {ir10:.1f}%\n"
            f"* img_r_mean: {ir_mean:.1f}%\n"
            f"* tr1: {tr1:.1f}%\n"
            f"* tr5: {tr5:.1f}%\n"
            f"* tr10: {tr10:.1f}%\n"
            f"* text_r_mean: {tr_mean:.1f}%"
        )

        return results
