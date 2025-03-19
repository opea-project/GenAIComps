import os.path as osp
import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import time
try:
    from clip import clip
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
except ImportError:
    from ..clip import clip
    from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'MiniImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = clip.build_model(state_dict or model.state_dict())
    #model.initialize_parameters()

    return model

    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        if self.cfg.TRAINER.COOP.XPU:
            prompts = prompts.to(self.cfg.TRAINER.COOP.XPU_ID)
        else:
            prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        #self.adapter = Adapter(512, 4).to(clip_model.dtype)

            
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
class CLIP_Bias(TrainerX):
    """ CLIP-Bias """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            #if 'adapter' not in name:
            param.requires_grad_(False)
        print('Turning on gradients in bias layer')
        print("=============")
        print(self.cfg.BIAS.BIAS_TERMS)
        print(self.cfg.BIAS.BIAS_TERMS_EXCLUDE)
        if len(self.cfg.BIAS.BIAS_TERMS) == 0 :
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    param.requires_grad_(True)
        else:
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    for str in self.cfg.BIAS.BIAS_TERMS:
                        if str in name:
                            param.requires_grad_(True)
        
        if len(self.cfg.BIAS.BIAS_TERMS_EXCLUDE) != 0 :
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    for str in self.cfg.BIAS.BIAS_TERMS_EXCLUDE:
                        if str in name:
                            param.requires_grad_(False)
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        
        if self.cfg.TRAINER.COOP.XPU:
            torch.xpu.set_device(self.cfg.TRAINER.COOP.XPU_ID)
            self.model.xpu(self.cfg.TRAINER.COOP.XPU_ID)
            if torch.xpu.device_count() > 1:
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                    device_ids=[self.cfg.TRAINER.COOP.XPU_ID])
        else:
            self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        print(self.optim)
        #print(self.optim)
        if self.cfg.TRAINER.COOP.XPU:
            import intel_extension_for_pytorch as ipex
            self.model, self.optim = ipex.optimize(model=self.model, optimizer=self.optim, level="O1")
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        #print(self.sched)
        
        self.register_model('clip_bias', self.model, self.optim, self.sched)
        # self.set_model_mode("train")
        # for name, param in self.model.named_parameters():
        #     print(name,param.requires_grad)
        device_count = torch.cuda.device_count()
        print("=====================================")
        print(device_count)
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            torch.cuda.set_device(self.cfg.TRAINER.COOP.CUDA_ID)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                    device_ids=[self.cfg.TRAINER.COOP.CUDA_ID])

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)     
        self.model_backward_and_update(loss)
        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            
            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
