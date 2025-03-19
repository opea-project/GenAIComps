import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from modeling.sampler import Sampler
from modeling.transformer import LayerNorm
from modeling.transformer import Transformer as TransformerClip
from modeling.clip_model import build_model, QuickGELU, Transformer

from utils.distributed import AllGather
allgather = AllGather.apply


class AdaCLIP(nn.Module):
    def __init__(self, cfg, clip_state_dict):
        super().__init__()

        self.num_frm = cfg.num_frm
        self.policy_backbone = cfg.policy_backbone
        self.use_policy = cfg.use_policy
        self.sim_header = cfg.sim_header
        self.word_agg = cfg.word_agg
        self.word_agg_temp = cfg.word_agg_temp
        self.frame_agg = cfg.frame_agg
        self.frame_agg_temp = cfg.frame_agg_temp
        self.reuse_scores = cfg.reuse_scores
        self.rank = cfg.rank
        self.world_size = cfg.world_size

        num_frm = cfg.top_k if self.use_policy else self.num_frm
        embed_dim = clip_state_dict["text_projection"].shape[1]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        context_length = clip_state_dict["positional_embedding"].shape[0]

        self.clip = build_model(clip_state_dict)
        if self.use_policy:
            self.sampler = Sampler(cfg, embed_dim)

        if self.sim_header == "seqTransf": # 4 layers following CLIP4Clip
            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
            self.transformerClip = TransformerClip(width=transformer_width, layers=4, heads=transformer_heads)
        elif self.sim_header == "transformer": # 1 layer
            self.frame_position_embeddings = nn.Embedding(num_frm, embed_dim)
            self.transformer = Transformer(width=transformer_width, layers=1, heads=transformer_heads)

        if self.frame_agg == "mlp":
            self.frame_agg_mlp = nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
                QuickGELU(),
                nn.Linear(2 * embed_dim, embed_dim),
                QuickGELU(),
                nn.Linear(embed_dim, 1)
            )
        elif self.frame_agg == "transformer":
            self.frame_agg_transformer = Transformer(width=transformer_width, layers=1, heads=transformer_heads)
            self.frame_agg_proj = nn.Linear(transformer_width, 1)

        if self.word_agg == "mlp":
            self.word_agg_mlp = nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
                QuickGELU(),
                nn.Linear(2 * embed_dim, embed_dim),
                QuickGELU(),
                nn.Linear(embed_dim, 1)
            )
        elif self.word_agg == "transformer":
            self.word_agg_transformer = TransformerClip(width=transformer_width, layers=1, heads=transformer_heads)
            self.word_agg_proj = nn.Linear(transformer_width, 1)

        self.initializer_range = 0.02
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, text_input_ids, clip_inputs, policy_inputs, tau=1, return_embds=False, gather=True):
        """
        text_input_ids: (B, num_caps, ctx_len)
        clip_inputs: (B, num_frm, C, H, W)
        policy_inputs: (B, num_frm, C, H, W)
        """
        frame_embd = self.get_visual_output(clip_inputs)
        return_hidden = self.word_agg
        text_embd, word_embd = self.get_text_output(text_input_ids, return_hidden)

        text_input_ids = rearrange(text_input_ids, "b n l -> (b n) l")
        lengths = text_input_ids.argmax(dim=-1) + 1

        if self.use_policy:
            policy_inputs = self.get_visual_output(clip_inputs) if self.policy_backbone == "clip" else policy_inputs
            actions, logits = self.sampler(policy_inputs, tau)
            frame_embd = actions @ frame_embd
            if self.reuse_scores:
                logits = actions @ logits.unsqueeze(-1)
                weights = F.softmax(logits / self.frame_agg_temp, dim=1)
                frame_embd = frame_embd * weights
            actions = actions.sum(dim=1)
        else:
            actions = torch.ones(clip_inputs.shape[:2]).to(clip_inputs.device)

        frame_embd = self.frame_transformation(frame_embd)

        if return_embds:
            return {
                "text_embd": text_embd,
                "frame_embd": frame_embd,
                "word_embd": word_embd if word_embd is not None else None,
                "actions": actions,
                "lengths": lengths,
            }
        else:
            if gather:
                text_embd = allgather(text_embd, self.rank, self.world_size)
                frame_embd = allgather(frame_embd, self.rank, self.world_size)
                word_embd = allgather(word_embd, self.rank, self.world_size) if word_embd is not None else None
                actions = allgather(actions, self.rank, self.world_size)
                lengths = allgather(lengths, self.rank, self.world_size)
                torch.distributed.barrier()
            sim_matrix = self.compute_sim_matrix(frame_embd, text_embd, word_embd, lengths)
            return sim_matrix, actions

    def get_text_output(self, text_input_ids, return_hidden=False):
        b = text_input_ids.shape[0]
        text_input_ids = rearrange(text_input_ids, "b n l -> (b n) l")
        text_embd, word_embd = self.clip.encode_text(text_input_ids, return_hidden)
        text_embd = rearrange(text_embd, "(b n) d -> b n d", b=b).float()
        if return_hidden:
            word_embd = rearrange(word_embd, "(b n) l d -> b n l d", b=b).float()
        return text_embd, word_embd

    def get_visual_output(self, visual_inputs):
        b = visual_inputs.shape[0]
        visual_inputs = rearrange(visual_inputs, "b n c h w -> (b n) c h w")
        frame_embd = self.clip.encode_image(visual_inputs).float()
        frame_embd = rearrange(frame_embd, "(b n) d -> b n d", b=b)
        return frame_embd

    def frame_transformation(self, frame_embd):
        if self.sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            frame_embd_original = frame_embd
            seq_length = frame_embd.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=frame_embd.device)
            position_ids = position_ids.unsqueeze(0).expand(frame_embd.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            frame_embd = frame_embd + frame_position_embeddings

            video_mask = torch.ones(frame_embd.shape[:2]).to(frame_embd.device)
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            frame_embd = frame_embd.permute(1, 0, 2)  # NLD -> LND
            frame_embd = self.transformerClip(frame_embd, extended_video_mask)
            frame_embd = frame_embd.permute(1, 0, 2)  # LND -> NLD
            frame_embd = frame_embd + frame_embd_original
        elif self.sim_header == "transformer":
            seq_length = frame_embd.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=frame_embd.device)
            position_ids = position_ids.unsqueeze(0).expand(frame_embd.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            frame_embd = frame_embd + frame_position_embeddings

            frame_embd = frame_embd.permute(1, 0, 2)  # NLD -> LND
            frame_embd = self.transformer(frame_embd)
            frame_embd = frame_embd.permute(1, 0, 2)  # LND -> NLD
        return frame_embd

    def compute_sim_matrix(self, frame_embd, text_embd, word_embd, lengths):
        if self.frame_agg:
            sim_matrix = self.frame_agg_sim_matrix(frame_embd, text_embd)
        elif self.word_agg:
            sim_matrix = self.word_agg_sim_matrix(frame_embd, word_embd, lengths)
        elif self.reuse_scores:
            sim_matrix = self.reuse_score_sim_matrix(frame_embd, text_embd)
        else:
            sim_matrix = self.mean_pool_sim_matrix(frame_embd, text_embd)
        return sim_matrix

    def mean_pool_sim_matrix(self, frame_embd, text_embd):
        text_embd = rearrange(text_embd, "b n d -> (b n) d")
        text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)
        frame_embd = frame_embd / frame_embd.norm(dim=-1, keepdim=True) 
        video_embd = frame_embd.mean(dim=1)
        video_embd = video_embd / video_embd.norm(dim=-1, keepdim=True)
        sims = self.clip.logit_scale.exp() * torch.matmul(text_embd, video_embd.t())
        return sims

    def reuse_score_sim_matrix(self, frame_embd, text_embd):
        text_embd = rearrange(text_embd, "b n d -> (b n) d")
        text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)
        video_embd = frame_embd.sum(dim=1)
        video_embd = video_embd / video_embd.norm(dim=-1, keepdim=True)
        sims = self.clip.logit_scale.exp() * torch.matmul(text_embd, video_embd.t())
        return sims

    def word_agg_sim_matrix(self, frame_embd, word_embd, lengths):
        b = word_embd.shape[0]
        word_embd = rearrange(word_embd, "b n l d -> (b n) l d")
        word_embd_orig = word_embd
        if self.word_agg == "mlp":
            logits = self.word_agg_mlp(word_embd)
        elif self.word_agg == "transformer":
            text_mask = torch.ones(word_embd.shape[:2]).to(word_embd.device) # 1s for tokens
            for i in range(lengths.shape[0]):
                text_mask[i, lengths[i]:] = 0
            extended_text_mask = (1.0 - text_mask.unsqueeze(1)) * -1000000.0
            extended_text_mask = extended_text_mask.expand(-1, text_mask.size(1), -1)
            word_embd = word_embd.permute(1, 0, 2)
            word_embd = self.word_agg_transformer(word_embd, extended_text_mask)
            word_embd = word_embd.permute(1, 0, 2)
            logits = self.word_agg_proj(word_embd)
        text_embd = []
        for i, x_len in enumerate(lengths):
            att_weights = F.softmax(logits[i, :x_len] / self.word_agg_temp, dim=0)
            text_embd.append((word_embd_orig[i, :x_len] * att_weights).sum(dim=0))
        text_embd = torch.stack(text_embd)
        text_embd = rearrange(text_embd, "(b n) d -> b n d", b=b)
        sims = self.mean_pool_sim_matrix(frame_embd, text_embd)
        return sims

    def frame_agg_sim_matrix(self, frame_embd, text_embd):
        if self.frame_agg == "mlp":
            sims = self.mlp_sim_matrix(frame_embd, text_embd)
        elif self.frame_agg == "transformer":
            sims = self.transformer_sim_matrix(frame_embd, text_embd)
        elif self.frame_agg == "qscore":
            sims = self.qscore_sim_matrix(frame_embd, text_embd)
        return sims

    def mlp_sim_matrix(self, frame_embd, text_embd):
        text_embd = rearrange(text_embd, "b n d -> (b n) d")
        text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)
        logits = self.frame_agg_mlp(frame_embd)
        weights = F.softmax(logits / self.frame_agg_temp, dim=1)
        video_embd = (frame_embd * weights).sum(dim=1)
        video_embd = video_embd / video_embd.norm(dim=-1, keepdim=True)
        sims = self.clip.logit_scale.exp() * torch.matmul(text_embd, video_embd.t())
        return sims

    def transformer_sim_matrix(self, frame_embd, text_embd):
        text_embd = rearrange(text_embd, "b n d -> (b n) d")
        text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)
        frame_embd_orig = frame_embd
        frame_embd = frame_embd.permute(1, 0, 2)
        frame_embd = self.frame_agg_transformer(frame_embd)
        frame_embd = frame_embd.permute(1, 0, 2)
        logits = self.frame_agg_proj(frame_embd)
        weights = F.softmax(logits / self.frame_agg_temp, dim=1)
        video_embd = (frame_embd_orig * weights).sum(dim=1)
        video_embd = video_embd / video_embd.norm(dim=-1, keepdim=True)
        sims = self.clip.logit_scale.exp() * torch.matmul(text_embd, video_embd.t())
        return sims

    def qscore_sim_matrix(self, frame_embd, text_embd, eps=1e-8):
        """
        Adapted from Q-Score: https://github.com/m-bain/clip-hitchhiker/blob/main/query_scoring.py#L4
        """
        text_embd = rearrange(text_embd, "b n d -> (b n) d")
        text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)
        frame_embd_orig = frame_embd
        frame_embd = frame_embd / frame_embd.norm(dim=-1, keepdim=True)
        sims = torch.einsum('ad, bvd -> abv', text_embd, frame_embd) # Bt, Bv, N
        weights = F.softmax(sims / self.frame_agg_temp, dim=2)
        frame_embd_weighted = torch.einsum('b v d, a b v -> a b v d', frame_embd_orig, weights)  # Bt, Bv, N, D
        video_embd = frame_embd_weighted.sum(dim=2) # Bt, Bv, D
        video_embd = video_embd / video_embd.norm(dim=2, keepdim=True)
        sims = self.clip.logit_scale.exp() * torch.einsum('a d, a b d -> a b', text_embd, video_embd)
        return sims

    def freeze_clip(self):
        for _, p in self.clip.named_parameters():
            p.requires_grad = False
