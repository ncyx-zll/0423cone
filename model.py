# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from cone.span_utils import generalized_temporal_iou, span_cxw_to_xx
from cone.matcher import build_matcher
from cone.transformer import build_transformer
from cone.position_encoding import build_position_encoding
from cone.misc import accuracy
# 【新增】：导入扩散相关组件
from cone_diffusion.denoising import GenerateCDNQueries

class CONE(nn.Module):
    """ This is the CONE model that performs moment localization in the long-form video. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_motion_dim, vid_appear_dim,
                 num_queries, input_dropout, aux_loss=False,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2,
                 adapter_module="linear", infer_query_init="coarse_logits"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_motion_dim: int, video visual motion feature input dimension
            vid_appear_dim: int, video visual appearance feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            adapter_module: str, one of [linear, mlp, none]
                linear: additional 2-layer MLP adapter (original)
                mlp:    stronger 2-layer MLP with LayerNorm + ReLU + residual
                        (better non-linear alignment, lower adapter_loss)
                none:   no adapter
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.infer_query_init = infer_query_init
        self.infer_t_mode = "single"
        self.infer_t_k = 20
        self.infer_sample_steps = 10
        self.infer_coarse_start_t = 30
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_txt_proj = nn.Sequential(*[
                                                 LinearLayer(txt_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[0]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[1]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[2])
                                             ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
                                                 LinearLayer(vid_motion_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[0]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[1]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[2])
                                             ][:n_input_proj])

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss
        # ====================================================================
        # 【新增】：初始化扩散核心组件
        # 1. 噪声生成器 (用于训练时从 GT 生成 noised reference points)
        self.dn_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=2,  # 0: foreground, 1: background
            label_embed_dim=hidden_dim,
            denoising_nums=100,  # 论文提到的 100 步训练扩散调度
            label_noise_prob=0.5,
            box_noise_scale=0.4,
        )
        # ====================================================================
        # 【补充修复 1】：需要将 0~100 的离散扩散步数 t，映射为连续的隐藏层特征
        self.t_embedder = nn.Embedding(self.dn_generator.denoising_nums, hidden_dim)

        # 【补充修复 2】：带噪的提议框是 2维 的 [t_center, d]，而 Transformer 需要 hidden_dim(比如256)维。
        # 我们用一个简单的 MLP 将 1D 时序坐标映射为高维的位置编码 (Positional Query)
        self.ref_point_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # =====
        self.adapter_module = adapter_module
        if self.adapter_module == "linear":
            self.adapter_layer = MLP(vid_appear_dim, hidden_dim, vid_appear_dim, 2)
        elif self.adapter_module == "mlp":
            # 更强的 MLP Adapter：LayerNorm → Linear → ReLU → Linear → LayerNorm → 残差
            # 输入输出双 LayerNorm：防止特征幅值爆炸，避免 NCE adapter_loss 出现 NaN
            # 相比 linear（仅2层线性），增加归一化和激活函数，提升非线性表示对齐能力
            self.adapter_layer = nn.Sequential(
                nn.LayerNorm(vid_appear_dim),      # 输入归一化
                nn.Linear(vid_appear_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, vid_appear_dim),
                nn.LayerNorm(vid_appear_dim),      # 输出归一化：防止幅值过大 → NCE NaN
            )

    def forward(self, src_txt, src_txt_mask, src_vid_motion, src_vid_motion_mask, gt_spans=None):
        """
        The forward expects two tensors:
           - src_txt: [batch_size, L_txt, D_txt]
           - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer
           - src_vid_motion: [batch_size, L_vid, D_vid]
           - src_vid_motion_mask: [batch_size, L_vid], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer

        It returns a dict with the following elements:
           - "pred_spans": The normalized boxes coordinates for all queries, represented as
                           (center_x, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        src_vid = self.input_vid_proj(src_vid_motion)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_motion_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_motion_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        batch_size = src.shape[0]
        device = src.device
        padding_mask = ~mask

        # Training path: coarse prediction first, then diffusion-refined prediction.
        if self.training:
            src_seq = src.permute(1, 0, 2)
            pos_seq = pos.permute(1, 0, 2)
            memory_seq = self.transformer.encoder(src_seq, src_key_padding_mask=padding_mask, pos=pos_seq)

            # 1) Coarse branch from static DETR queries.
            coarse_query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
            coarse_tgt = torch.zeros_like(coarse_query_pos)
            coarse_hs = self.transformer.decoder(
                coarse_tgt,
                memory_seq,
                memory_key_padding_mask=padding_mask,
                pos=pos_seq,
                query_pos=coarse_query_pos,
                time_embed=None,
            ).transpose(1, 2)
            coarse_class = self.class_embed(coarse_hs)
            coarse_coord = self.span_embed(coarse_hs)
            coarse_coord_for_out = coarse_coord.sigmoid() if self.span_loss_type == "l1" else coarse_coord

            # 2) Refined branch from diffusion noised box queries.
            if gt_spans is not None:
                gt_boxes_list = [t['spans'] for t in gt_spans]
                gt_labels_list = [torch.zeros(len(b), dtype=torch.long, device=device) for b in gt_boxes_list]
                _, noised_box_queries, _, _, _ = self.dn_generator(gt_labels_list, gt_boxes_list)
                t = torch.randint(0, self.dn_generator.denoising_nums, (batch_size,), device=device).long()
            else:
                noised_box_queries = torch.randn(batch_size, self.num_queries, 2, device=device)
                t = torch.zeros(batch_size, device=device).long()

            time_embed_features = self.t_embedder(t).unsqueeze(1).expand(-1, self.num_queries, -1)
            refine_query_pos = self.ref_point_embed(noised_box_queries).permute(1, 0, 2)
            refine_tgt = torch.zeros_like(refine_query_pos)
            refine_hs = self.transformer.decoder(
                refine_tgt,
                memory_seq,
                memory_key_padding_mask=padding_mask,
                pos=pos_seq,
                query_pos=refine_query_pos,
                time_embed=time_embed_features,
            ).transpose(1, 2)

            refine_class = self.class_embed(refine_hs)
            refine_coord = self.span_embed(refine_hs)
            refine_coord_for_out = refine_coord.sigmoid() if self.span_loss_type == "l1" else refine_coord

            # Keep pred_spans/pred_logits as refined outputs for main criterion path.
            out = {
                'pred_logits': refine_class[-1],
                'pred_spans': refine_coord_for_out[-1],
                'pred_logits_coarse': coarse_class[-1],
                'pred_spans_coarse': coarse_coord_for_out[-1],
                'pred_logits_refined': refine_class[-1],
                'pred_spans_refined': refine_coord_for_out[-1],
            }
            if gt_spans is not None:
                out['denoising_groups'] = True

            memory = memory_seq.transpose(0, 1)
            txt_mem = memory[:, src_vid.shape[1]:]
            vid_mem = memory[:, :src_vid.shape[1]]
            out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)

            if self.aux_loss:
                out['aux_outputs'] = []
                # 1) 监督 Coarse 分支的所有层，让 coarse 初始化具备可学习定位能力。
                for a, b in zip(coarse_class, coarse_coord_for_out):
                    out['aux_outputs'].append({'pred_logits': a, 'pred_spans': b})

                # 2) 监督 Refine 分支中间层（最后一层已由主分支 pred_* 监督）。
                for a, b in zip(refine_class[:-1], refine_coord_for_out[:-1]):
                    out['aux_outputs'].append({'pred_logits': a, 'pred_spans': b})
            return out

        # ======================================================================
        # 推理 / 验证分支（eval/inference）：扩散迭代采样 (Iterative Denoising Sampling)
        #
        # 核心目标：
        #   - 复用一次 Encoder 计算得到的 memory（视频+文本融合特征），避免每步重复编码。
        #   - Decoder 进行多步迭代，逐步把 noisy 的 span logits 还原成干净的 span。
        #
        # 两种初始化策略：
        #   1) infer_query_init == "randn"
        #      - 经典“文生图”式采样：从纯高斯噪声（无界 logit 空间）开始，倒推完整 T=100 步调度。
        #   2) infer_query_init == "coarse_logits"
        #      - “图生图/截断扩散”式采样：
        #        先用标准 DETR query 做一次 coarse 前向得到较合理的 span logits，再从较小 start_t
        #        （例如 30）开始倒推。并在进入循环前，主动施加与 start_t 对应强度的前向噪声，
        #        使输入分布与训练时的 noisy state 对齐。
        # ======================================================================

        # 1) Encoder：将 (视频特征 + 文本特征) 编码为 memory（供 Decoder 跨注意力检索）
        # src: [B, L, D] -> [L, B, D] 以适配 torch.nn.MultiheadAttention 的序列优先格式
        src_seq = src.permute(1, 0, 2)
        pos_seq = pos.permute(1, 0, 2)
        # memory_seq: [L, B, D]
        memory_seq = self.transformer.encoder(src_seq, src_key_padding_mask=padding_mask, pos=pos_seq)

        # 2) 时间步调度：支持 multi/single/double 三种推理消融模式
        sample_steps = max(1, int(self.infer_sample_steps))
        max_t = self.dn_generator.denoising_nums

        def build_timesteps(start_t):
            start_t = int(start_t)
            if start_t <= 0:
                raise ValueError(f"Invalid start_t={start_t}.")
            if self.infer_t_mode == "single":
                return [start_t - 1]
            if self.infer_t_mode == "double":
                k = int(self.infer_t_k)
                k = max(0, min(k, max_t - 1))
                return [k] if k == 0 else [k, 0]
            step_size = max(1, start_t // sample_steps)
            timesteps = list(range(start_t - 1, -1, -step_size))
            if len(timesteps) == 0 or timesteps[-1] != 0:
                timesteps.append(0)
            return timesteps

        start_t = max_t
        timesteps = None

        if self.infer_query_init == "coarse_logits" and self.span_loss_type == "l1":
            # --------------------------------------------------------------
            # (A) coarse_logits 初始化：先跑一次标准 DETR coarse 解码
            # --------------------------------------------------------------
            # coarse_query_pos: [Q, B, D]，来自可学习的 query embedding
            coarse_query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
            coarse_tgt = torch.zeros_like(coarse_query_pos)
            # coarse_hs: [#layers, B, Q, D]
            coarse_hs = self.transformer.decoder(
                coarse_tgt,
                memory_seq,
                memory_key_padding_mask=padding_mask,
                pos=pos_seq,
                query_pos=coarse_query_pos,
                time_embed=None,
            ).transpose(1, 2)

            # coarse_class: [B, Q, 2]，分类 logits（0: foreground, 1: background）
            coarse_class = self.class_embed(coarse_hs)[-1]
            # coarse_coord: [B, Q, 2]，span logits（注意：这里还没 sigmoid，处于无界 logit 空间）
            coarse_coord = self.span_embed(coarse_hs)[-1]

            # 用前景概率排序，让“更像目标”的 query 排在前面（不改变 Q 数量，只重排）
            fg_scores = coarse_class.softmax(-1)[..., 0]
            sort_idx = torch.argsort(fg_scores, dim=1, descending=True)
            gather_idx = sort_idx.unsqueeze(-1).expand(-1, -1, coarse_coord.shape[-1])
            box_queries = torch.gather(coarse_coord, dim=1, index=gather_idx)

            # coarse 初始化可从截断时间步开始，多步模式默认 start_t=30。
            start_t = int(self.infer_coarse_start_t) if self.infer_t_mode == "multi" else max_t
            start_t = max(1, min(start_t, max_t))
            timesteps = build_timesteps(start_t)

            # --------------------------------------------------------------
            # (C) 注入前向噪声（关键）：让 coarse logits 对齐到“t=start_t 的 noisy state”分布
            #
            # 训练阶段在 denoising.py 中：
            #   - 对 [center, width] 在 [0,1] 空间加扰动并 clamp
            #   - 再 inverse_sigmoid 映射到 logit 空间，成为 noised_box_queries
            #
            # 这里我们反过来：
            #   - coarse_logits -> sigmoid 得到 [0,1] span
            #   - 参考 apply_box_noise 的最大偏移量定义，用 U(-1,1) 噪声按强度缩放
            #   - 再 clamp 回 [0,1] 并 inverse_sigmoid 回到 logit 空间
            #
            # 注意：这里的噪声强度是一个启发式线性缩放：scale = box_noise_scale * start_t / T
            # 这样 start_t 越大，噪声越大；start_t 越小，噪声越弱。
            # --------------------------------------------------------------
            from cone_diffusion.denoising import inverse_sigmoid

            first_step = int(timesteps[0])
            if first_step > 0:
                with torch.no_grad():
                    # [B, Q, 2] in [0,1]
                    coarse_spans_01 = box_queries.sigmoid().clamp(0.0, 1.0)
                    diff = torch.zeros_like(coarse_spans_01)
                    diff[:, :, :1] = coarse_spans_01[:, :, 1:] / 2
                    diff[:, :, 1:] = coarse_spans_01[:, :, 1:]
                    # 让噪声强度和首个去噪步大致一致
                    noise_strength = float(self.dn_generator.box_noise_scale) * (first_step / float(max_t))
                    coarse_spans_01 = coarse_spans_01 + (torch.rand_like(coarse_spans_01) * 2 - 1.0) * diff * noise_strength
                    coarse_spans_01 = coarse_spans_01.clamp(0.0, 1.0)
                    box_queries = inverse_sigmoid(coarse_spans_01)

        else:
            # randn 初始化：直接从无界高斯噪声开始（天然与 logit 空间对齐）
            box_queries = torch.randn(batch_size, self.num_queries, 2, device=device)
            timesteps = build_timesteps(max_t)

        if timesteps is None:
            timesteps = build_timesteps(max_t)

        outputs_class = None
        outputs_coord = None
        for step in timesteps:
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            time_embed_features = self.t_embedder(t).unsqueeze(1).expand(-1, self.num_queries, -1)

            query_pos = self.ref_point_embed(box_queries).permute(1, 0, 2)
            tgt = torch.zeros_like(query_pos)
            hs = self.transformer.decoder(
                tgt,
                memory_seq,
                memory_key_padding_mask=padding_mask,
                pos=pos_seq,
                query_pos=query_pos,
                time_embed=time_embed_features,
            )
            hs = hs.transpose(1, 2)

            outputs_class = self.class_embed(hs)
            outputs_coord = self.span_embed(hs)
            box_queries = outputs_coord[-1]

        if outputs_class is None or outputs_coord is None:
            raise RuntimeError("No decoder outputs produced in eval denoising loop. Check denoising timestep setup.")

        final_spans = box_queries.sigmoid() if self.span_loss_type == "l1" else box_queries
        out = {'pred_logits': outputs_class[-1], 'pred_spans': final_spans}

        memory = memory_seq.transpose(0, 1)
        txt_mem = memory[:, src_vid.shape[1]:]
        vid_mem = memory[:, :src_vid.shape[1]]
        out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)

        if self.aux_loss:
            if self.span_loss_type == "l1":
                aux_spans = [b.sigmoid() for b in outputs_coord[:-1]]
            else:
                aux_spans = list(outputs_coord[:-1])
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], aux_spans)
            ]

        return out

    def forward_clip_matching(self, src_cls_txt, src_vid_appear, src_vid_appear_mask, proposal=None,
                              is_groundtruth=False):
        """
        The forward expects following tensors:
            - src_cls_txt: [batch_size, D_txt]
            - src_vid_appear: [batch_size, L_vid, D_vid]
            - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
            - proposal:
            - is_groundtruth: whether the proposal comes from the ground-truth (during training)
            or proposal generation prediction (during inference).
        It returns a proposal-query similarity matrix.
        """
        text_cls_features = src_cls_txt / src_cls_txt.norm(dim=1, keepdim=True)

        if is_groundtruth:
            tgt_proposals = torch.vstack([t["proposal"][0] for t in proposal])  # (#spans, 2)
            proposal_feat = self._get_groundtruth_proposal_feat(src_vid_appear, tgt_proposals)
            proposal_features = proposal_feat / proposal_feat.norm(dim=1, keepdim=True)
            return torch.einsum('bd,ad->ba', proposal_features, text_cls_features)
        else:
            proposal_feat = self._get_predicted_proposal_feat(src_vid_appear, src_vid_appear_mask, proposal)
            proposal_features = proposal_feat / proposal_feat.norm(dim=2, keepdim=True)
            return torch.einsum('bld,bd->bl', proposal_features, text_cls_features)

    def _get_groundtruth_proposal_feat(self, src_vid_appear, groundtruth_proposal):
        """
        The forward expects following tensors:
           - src_vid_appear: [batch_size, L_vid, D_vid]
           - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
           - proposal: [batch_size, 2], ground-truth start and end timestamps
       It returns proposal features for ground-truth moments.
        """
        proposal_feat_list = []
        for idx, (feat, start_end_list) in enumerate(zip(src_vid_appear, groundtruth_proposal)):
            clip_feat = feat[start_end_list[0]:start_end_list[1]]
            # mean pooling inside each proposal
            # 防止空区间（start>=end）时 mean() 返回 NaN
            if clip_feat.shape[0] == 0:
                proposal_feat_list.append(torch.zeros(feat.shape[-1], device=feat.device, dtype=feat.dtype))
            else:
                proposal_feat_list.append(clip_feat.mean(axis=0))

        proposal_feat = torch.vstack(proposal_feat_list)

        # adapter module
        if self.adapter_module in ("linear", "mlp"):
            proposal_feat = self.adapter_layer(proposal_feat) + proposal_feat
        else:
            proposal_feat = proposal_feat

        return proposal_feat

    def _get_predicted_proposal_feat(self, src_vid_appear, src_vid_appear_mask, pred_proposal):
        """
        The forward expects following tensors:
          - src_vid_appear: [batch_size, L_vid, D_vid]
          - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
          - proposal: [batch_size, N_query, 2], predicted start and end timestamps for each moment queries
        It returns proposal features for predicted proposals.
        """
        vid_appear_dim = src_vid_appear.shape[2]
        duration = torch.sum(src_vid_appear_mask, dim=-1)
        proposal = torch.einsum('bld,b->bld', span_cxw_to_xx(pred_proposal), duration)  # .to(torch.int32)

        bsz, n_query = proposal.shape[:2]
        proposal_start = F.relu(torch.floor(proposal[:, :, 0]).to(torch.int32))
        proposal_end = torch.ceil(proposal[:, :, 1]).to(torch.int32)

        proposal_feat_list = []
        for idx, (feat, start_list, end_list) in enumerate(zip(src_vid_appear, proposal_start, proposal_end)):
            for start, end in zip(start_list, end_list):
                clip_feat = feat[start:end]
                # mean pooling inside each proposal
                # 防止空区间（start>=end）时 mean() 返回 NaN
                if clip_feat.shape[0] == 0:
                    proposal_feat_list.append(torch.zeros(feat.shape[-1], device=feat.device, dtype=feat.dtype))
                else:
                    proposal_feat_list.append(clip_feat.mean(axis=0))
        proposal_feat = torch.vstack(proposal_feat_list)

        # adapter module
        if self.adapter_module in ("linear", "mlp"):
            proposal_feat = self.adapter_layer(proposal_feat) + proposal_feat
        else:
            proposal_feat = proposal_feat

        proposal_feat = proposal_feat.reshape(bsz, n_query, vid_appear_dim)

        return proposal_feat


class SetCriterion(nn.Module):
    """ This class computes the loss of CONE modified from DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model inside the positive window
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, match_span_key="pred_spans_refined", debug_match=False):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        self.match_span_key = match_span_key
        self.debug_match = debug_match
        self._printed_match_debug = False

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_adapter(self, pos_outputs):
        ######
        # additional adapter NCE loss, followed by CLIP implementation
        #####
        assert 'logits_per_video' in pos_outputs

        logits_per_video = pos_outputs["logits_per_video"] / self.temperature
        bsz = len(logits_per_video)
        diagonal_indices = torch.arange(bsz).to(logits_per_video.device)

        criterion = nn.CrossEntropyLoss(reduction="mean")
        loss_per_video = criterion(logits_per_video, diagonal_indices)
        loss_per_text = criterion(logits_per_video.T, diagonal_indices)
        loss = (loss_per_video + loss_per_text) / 2
        return {'loss_adapter': loss}

    def loss_spans(self, outputs, targets, indices, neg_outputs=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        ######
        # no modification
        #####
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {'loss_span': loss_span.mean(), 'loss_giou': loss_giou.mean()}
        return losses

    def loss_labels(self, outputs, targets, indices=None, neg_outputs=None, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch

        ######
        # additional contrastive classification loss to distinguish positive and negative window via proposal-level comparison
        # summary: assign all proposal label in the negative window is the background label
        ######
        if neg_outputs is not None:
            neg_src_logits = neg_outputs['pred_logits']  # (batch_size, #queries, #classes=2)
            src_logits = torch.cat((src_logits, neg_src_logits), dim=1)
        #####

        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        if indices is not None:
            idx = self._get_src_permutation_idx(indices)
            target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if indices is not None and bool(log):
            # TODO this should probably be a separate loss, not hacked in this one here
            # Ensure `idx` is initialized before use
            if indices is not None:
                idx = self._get_src_permutation_idx(indices)
                losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
            else:
                # Fix type mismatch for `class_error` assignment
                losses['class_error'] = torch.tensor(100, dtype=torch.float32)
        return losses

    def loss_saliency(self, outputs, targets, indices, neg_outputs=None):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        ######
        # original saliency loss for Moment-DETR
        # summary: saliency score of random frame inside the ground-truth  is larger than that outside of ground-truth
        ######
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or motion_window_80
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        ######
        # additional contrastive saliency loss to distinguish positive and negative window via saliency-level comparison
        # summary: saliency score of random frame inside the ground-truth  is larger than that the maximum saliency score of the negative window
        ######
        if neg_outputs is not None:
            neg_saliency_scores = neg_outputs["saliency_scores"]  # (N, L)
            neg_saliency_max_scores, _ = torch.max(neg_saliency_scores, 1)
            neg_window_max_scores = torch.stack(
                [neg_saliency_max_scores for _ in range(num_pairs)], dim=1)
            loss_neg_saliency = torch.clamp(self.saliency_margin + neg_window_max_scores - pos_scores, min=0).sum() \
                                / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
            loss_saliency += loss_neg_saliency

        return {"loss_saliency": loss_saliency}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, neg_outputs, **kwargs):
        loss_map = {
            "spans": lambda outputs, targets, indices, neg_outputs=None: self.loss_spans(outputs, targets, indices, neg_outputs),
            "labels": lambda outputs, targets, indices, neg_outputs=None, log=True: self.loss_labels(outputs, targets, indices, neg_outputs, log=log),
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # Ensure `kwargs` are filtered to match the expected parameters of the loss functions
        if loss == "labels":
            kwargs = {k: v for k, v in kwargs.items() if k in ["log"]}
        return loss_map[loss](outputs, targets, indices, neg_outputs, **kwargs)

    def forward(self, outputs, targets, neg_outputs=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts for positive window, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             neg_outputs: list of dicts for negative window, such that len(neg_outputs) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        if targets is None:
            losses = {}
            losses.update(self.loss_labels(outputs, targets, None))
            return losses

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        # indices = self.matcher(outputs_without_aux, targets)
        # Compute all the requested losses

        # ====================================================================
        # 【架构纠正】：恢复完整的 DETR 行为
        # 无论提议是否经过扩散去噪，最终的预测集合都必须通过匈牙利算法进行全局最优匹配
        # ====================================================================
        indices = self.matcher(outputs_without_aux, targets, span_key=self.match_span_key)
        if self.debug_match and not self._printed_match_debug:
            match_sizes = [int(len(src_idx)) for src_idx, _ in indices]
            sample_src = indices[0][0][:5].tolist() if len(indices) > 0 else []
            sample_tgt = indices[0][1][:5].tolist() if len(indices) > 0 else []
            print(
                f"[DEBUG][SetCriterion] span_key={self.match_span_key}; "
                f"matched_per_batch={match_sizes}; sample0_src={sample_src}; sample0_tgt={sample_tgt}"
            )
            self._printed_match_debug = True

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, neg_outputs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                # 【架构纠正】：辅助层同样进行严格的匈牙利匹配
                indices = self.matcher(aux_outputs, targets, span_key="pred_spans")
                for loss in self.losses:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, neg_outputs, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = CONE(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_motion_dim=args.v_motion_feat_dim,
        vid_appear_dim=args.v_appear_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        span_loss_type=args.span_loss_type,
        adapter_module=args.adapter_module,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        infer_query_init=getattr(args, "infer_query_init", "coarse_logits"),
    )
    model.infer_t_mode = getattr(args, "infer_t_mode", "single")
    model.infer_t_k = getattr(args, "infer_t_k", 20)
    model.infer_sample_steps = getattr(args, "infer_sample_steps", 10)
    model.infer_coarse_start_t = getattr(args, "infer_coarse_start_t", 30)

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.adapter_loss:
        weight_dict["loss_adapter"] = args.adapter_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
        match_span_key="pred_spans_refined",
        debug_match=args.debug,
    )
    criterion.to(device)
    return model, criterion

