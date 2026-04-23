# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

# from detrex.utils import inverse_sigmoid
# ==========================================================
# 【新增】：替代 Detrex 的原生 inverse_sigmoid 实现
def inverse_sigmoid(x, eps=1e-5):
    """
    将输入张量限制在 (0, 1) 区间，并计算其反 Sigmoid 值。
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
# ==========================================================

def apply_label_noise(
    labels: torch.Tensor,
    label_noise_prob: float = 0.2,
    num_classes: int = 80,
):
    """
    Args:
        labels (torch.Tensor): Classification labels with ``(num_labels, )``.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        num_classes (int): Number of total categories.

    Returns:
        torch.Tensor: The noised labels the same shape as ``labels``.
    """
    if label_noise_prob > 0:
        p = torch.rand_like(labels.float())
        noised_index = torch.nonzero(p < label_noise_prob).view(-1)
        new_lebels = torch.randint_like(noised_index, 0, num_classes)
        noised_labels = labels.scatter_(0, noised_index, new_lebels)
        return noised_labels
    else:
        return labels


def apply_box_noise(  #修改为1D
    boxes: torch.Tensor,
    box_noise_scale: float = 0.4,
):
    """
    Args:
        boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
            shape ``(num_boxes, 4)``
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
    """
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        # diff[:, :2] = boxes[:, 2:] / 2
        # diff[:, 2:] = boxes[:, 2:]
        # 对于 1D，boxes[:, 1:] 是时长 d
        # t_center 的最大偏移量应该是 d/2，d 的最大偏移量是 d
        diff[:, :1] = boxes[:, 1:] / 2
        diff[:, 1:] = boxes[:, 1:]
        # 施加均匀分布噪声 (-1 到 1) 并缩放
        boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
        # 严格限制在 [0, 1] 局部窗口内！
        boxes = boxes.clamp(min=0.0, max=1.0)
    return boxes


class GenerateDNQueries(nn.Module):
    """Generate denoising queries for DN-DETR

    Args:
        num_queries (int): Number of total queries in DN-DETR. Default: 300
        num_classes (int): Number of total categories. Default: 80.
        label_embed_dim (int): The embedding dimension for label encoding. Default: 256.
        denoising_groups (int): Number of noised ground truth groups. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4
        with_indicator (bool): If True, add indicator in noised label/box queries.

    """

    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = False,
    ):
        super(GenerateDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator

        # leave one dim for indicator mentioned in DN-DETR
        if with_indicator:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def generate_query_masks(self, max_gt_num_per_image, device):
        noised_query_nums = max_gt_num_per_image * self.denoising_groups
        tgt_size = noised_query_nums + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.denoising_groups):
            if i == 0:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    max_gt_num_per_image * (i + 1) : noised_query_nums,
                ] = True
            if i == self.denoising_groups - 1:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    : max_gt_num_per_image * i,
                ] = True
            else:
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    max_gt_num_per_image * (i + 1) : noised_query_nums,
                ] = True
                attn_mask[
                    max_gt_num_per_image * i : max_gt_num_per_image * (i + 1),
                    : max_gt_num_per_image * i,
                ] = True
        return attn_mask

    def forward(
        self,
        gt_labels_list,
        gt_boxes_list,
    ):
        """
        Args:
            gt_boxes_list (list[torch.Tensor]): Ground truth bounding boxes per image
                with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)``
            gt_labels_list (list[torch.Tensor]): Classification labels per image in shape ``(num_gt, )``.
        """

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # e.g. tensor([0, 1, 2, 2, 3, 4]) -> tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]) if group = 2.
        gt_labels = gt_labels.repeat(self.denoising_groups, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # Add noise on labels and boxes
        noised_labels = apply_label_noise(gt_labels, self.label_noise_prob, self.num_classes)
        noised_boxes = apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]

        # add indicator to label encoding if with_indicator == True
        if self.with_indicator:
            label_embedding = torch.cat([label_embedding, torch.ones([query_num, 1]).to(device)], 1)

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # the total denoising queries is depended on denoising groups and max number of instances.
        noised_query_nums = max_gt_num_per_image * self.denoising_groups

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = (
            torch.zeros(noised_query_nums, self.label_embed_dim).to(device).repeat(batch_size, 1, 1)
        )
        noised_box_queries = torch.zeros(noised_query_nums, 2).to(device).repeat(batch_size, 1, 1)
        #原本是生成 4 维坐标，现在要改成 2 维
        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(
            batch_idx, torch.tensor(gt_nums_per_image).long()
        )

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat(
                [torch.tensor(list(range(num))) for num in gt_nums_per_image]
            )
            valid_index_per_group = torch.cat(
                [
                    valid_index_per_group + max_gt_num_per_image * i
                    for i in range(self.denoising_groups)
                ]
            ).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            max_gt_num_per_image,
        )


class GenerateCDNQueries(nn.Module):
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_nums: int = 100,
        label_noise_prob: float = 0.5,
        box_noise_scale: float = 1.0,
    ):
        super(GenerateCDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_nums = denoising_nums
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        
        self.label_encoder = nn.Embedding(num_classes, label_embed_dim)
    
    def forward(
        self,
        gt_labels_list,
        gt_boxes_list,
    ):
        if not isinstance(gt_labels_list, (list, tuple)) or not isinstance(gt_boxes_list, (list, tuple)):
            raise TypeError(
                "gt_labels_list and gt_boxes_list must be list/tuple with per-image tensors"
            )
        if gt_labels_list is None or gt_boxes_list is None:
            raise ValueError("gt_labels_list and gt_boxes_list must not be None")
        if len(gt_labels_list) != len(gt_boxes_list):
            raise ValueError(
                f"Mismatched batch lists: labels={len(gt_labels_list)}, boxes={len(gt_boxes_list)}"
            )
        if len(gt_labels_list) == 0:
            raise ValueError("Empty batch is not supported for denoising query generation")

        batch_size = len(gt_labels_list)
        denoising_groups = self.denoising_nums * 2

        # Resolve device/dtype from inputs if possible.
        device = torch.device("cpu")
        box_dtype = torch.float32
        for boxes in gt_boxes_list:
            if isinstance(boxes, torch.Tensor):
                device = boxes.device
                box_dtype = boxes.dtype
                break

        noised_label_queries = torch.zeros(
            batch_size, self.num_queries, self.label_embed_dim, device=device
        )
        noised_box_queries = torch.zeros(batch_size, self.num_queries, 2, device=device, dtype=box_dtype)

        gt_nums_per_image = []
        for b in range(batch_size):
            labels = gt_labels_list[b]
            boxes = gt_boxes_list[b]

            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=device, dtype=torch.long)
            else:
                labels = labels.to(device=device, dtype=torch.long)
            labels = labels.view(-1)

            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, device=device, dtype=box_dtype)
            else:
                boxes = boxes.to(device=device, dtype=box_dtype)

            if boxes.numel() == 0:
                gt_nums_per_image.append(0)
                # Empty GT fallback: use random normalized boxes then map to inverse-sigmoid space.
                sampled_boxes = torch.rand(self.num_queries, 2, device=device, dtype=box_dtype)
                sampled_labels = torch.zeros(self.num_queries, device=device, dtype=torch.long)
            else:
                if boxes.dim() == 1:
                    boxes = boxes.unsqueeze(0)
                elif boxes.dim() != 2:
                    raise ValueError(f"Expected boxes dim 1 or 2, got shape {tuple(boxes.shape)}")
                if boxes.shape[-1] > 2:
                    boxes = boxes[:, :2]
                elif boxes.shape[-1] < 2:
                    raise ValueError(f"Expected gt boxes last dim >= 2, got shape {tuple(boxes.shape)}")

                n = boxes.shape[0]
                gt_nums_per_image.append(n)
                repeat_factor = (self.num_queries + n - 1) // n
                sampled_boxes = boxes.repeat(repeat_factor, 1)[: self.num_queries]

                if labels.numel() == 0:
                    sampled_labels = torch.zeros(self.num_queries, device=device, dtype=torch.long)
                else:
                    labels = labels.clamp(min=0, max=self.num_classes - 1)
                    label_repeat = (self.num_queries + labels.numel() - 1) // labels.numel()
                    sampled_labels = labels.repeat(label_repeat)[: self.num_queries]

            sampled_labels = apply_label_noise(
                sampled_labels.clone(), self.label_noise_prob, self.num_classes
            )
            label_embedding = self.label_encoder(sampled_labels)

            sampled_boxes = apply_box_noise(sampled_boxes.clone(), self.box_noise_scale)
            sampled_boxes = inverse_sigmoid(sampled_boxes)

            noised_label_queries[b] = label_embedding
            noised_box_queries[b] = sampled_boxes

        max_gt_num_per_image = max(gt_nums_per_image) if gt_nums_per_image else 0
        attn_mask = torch.zeros(self.num_queries, self.num_queries, device=device, dtype=torch.bool)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            denoising_groups,
            max_gt_num_per_image,
        )
