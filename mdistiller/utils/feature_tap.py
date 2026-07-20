# -*- coding: utf-8 -*-
"""
文件名：mdistiller/utils/feature_tap.py

【用途概述】
    - 在“不改动模型 forward”的前提下，自动抓取“分类器前一层”的特征向量（通常是全局表征，维度合适且语义浓缩）。
    - 适配常见分类网络（ResNet / VGG / WRN / MobileNet / ViT 等）；核心做法是在“最后一个 nn.Linear”上挂 forward hook，
      在其前向时捕捉传入 Linear 的输入张量（即我们需要的 pre-classifier feature）。

【为什么这样做】
    - 你的仓库里“vanilla 教师”已经训练完毕，改动其 forward 可能较麻烦；
      用 hook 可以零侵入地得到特征，兼容历史代码。
    - hook 读取的是 Linear 的输入（仍在计算图中），因此对学生模型来说，梯度可以顺利反传至 backbone。

【主要接口】
    - attach_preclassifier_hook(model) -> (handle, get_feat_fn)
        * handle：hook 句柄，可在不需要时 .remove() 移除；
        * get_feat_fn()：一次前向执行后调用，返回刚刚捕获的特征张量（带计算图）。
    - FeatureTapper(model)：
        * 一个“轻量包装器”，让任意分类模型“看起来像”返回 (logits, feat)；
        * __call__(x) 直接得到 (logits, feat)。

【注意】
    - 该实现默认“最后一个 nn.Linear 即为分类头”，相当通用；
      若你的特定模型分类头不是 Linear（比如 ViT 的 head 另有封装），可以在 _find_last_linear 中改为匹配实际头部模块。
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional


def _find_last_linear(module: nn.Module) -> Optional[nn.Linear]:
    """
    在模型的所有子模块中顺序遍历，返回“最后出现的 nn.Linear 模块”。
    这在 ResNet/VGG/WRN/MobileNet 等大多数分类模型中即为分类器。
    """
    last_lin = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_lin = m
    return last_lin


def attach_preclassifier_hook(model: nn.Module) -> Tuple[torch.utils.hooks.RemovableHandle, Callable[[], torch.Tensor]]:
    """
    在模型中找到“最后的 nn.Linear”，注册 forward hook，捕捉其输入（pre-classifier feature）。

    参数：
        model: 任意分类模型实例

    返回：
        handle:  hook 句柄，可用于 handle.remove() 移除 hook；
        get_feat: 无参函数，需在一次 forward 后调用，用以取回本次前向捕获到的特征张量（仍保留计算图）。
    """
    target = _find_last_linear(model)
    if target is None:
        raise RuntimeError("未找到 nn.Linear 作为分类头；若你的模型不使用 Linear 作为分类头，请在此处自定义匹配规则。")

    cache = {"feat": None}

    def _hook(module, inputs, output):
        """
        forward hook 回调：
            inputs 是一个 tuple，其第 0 个元素就是传入 Linear 的特征向量（形状一般是 [N, D]）。
        """
        cache["feat"] = inputs[0]  # 保留在计算图里

    handle = target.register_forward_hook(_hook)

    def get_feat():
        assert cache["feat"] is not None, "尚未执行 forward 或未触发 hook，请先执行一次模型前向。"
        return cache["feat"]

    return handle, get_feat


class FeatureTapper(nn.Module):
    """
    轻量包装器：把原模型“包装”为返回 (logits, pre-classifier feat) 的可调用对象。
    用法示例：
        tap_t = FeatureTapper(teacher)
        logits, feat = tap_t(images)
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._handle, self._get_feat = attach_preclassifier_hook(self.model)

    def forward(self, x):
        """
        前向逻辑：
            1) 直接调用底层模型得到 logits（兼容 tensor/tuple/dict）；
            2) 从 hook 中读取 pre-classifier 特征。
        """
        out = self.model(x)

        # 兼容常见输出结构（tensor / tuple / dict）
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif isinstance(out, dict):
            logits = out.get("logits", list(out.values())[0])
        else:
            logits = out

        feat = self._get_feat()
        return logits, feat
