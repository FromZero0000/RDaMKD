# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


class NKDLoss(nn.Module):
    """
    Neighborhood Knowledge Distillation loss.

    Args:
        k (int): number of neighbors
        choose (str): neighbor selection strategy, currently supports "angle"
        lambda1 (float): weight for logits residual relation
        lambda2 (float): weight for feature neighborhood relation
    """

    def __init__(self, k=1, choose="angle", lambda1=10.0, lambda2=10.0):
        super(NKDLoss, self).__init__()
        self.k = k
        self.choose = choose
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def angle(self, t):
        """
        Use cosine similarity of teacher logits to build neighborhood.
        t: [B, C]
        """
        t = F.normalize(t, dim=1)
        cosine = torch.mm(t, t.t())  # [B, B]
        sim, idx = torch.sort(cosine, dim=1, descending=True)
        return sim, idx

    def js_div(self, q, p):
        """
        Jensen-Shannon divergence.
        q, p: [B, K, C]
        """
        q = F.softmax(q, dim=2)
        p = F.softmax(p, dim=2)

        mean_prob = 0.5 * (q + p)
        log_mean_prob = torch.log(mean_prob + 1e-12)

        loss = 0.5 * (
            F.kl_div(log_mean_prob, q, reduction="sum")
            + F.kl_div(log_mean_prob, p, reduction="sum")
        )
        return loss

    def fea(self, fea_s, fea_t, nebor_idx):
        """
        Feature-level neighborhood relation distillation.
        fea_s, fea_t: list of feature maps, each [B, C, H, W]
        nebor_idx: [B, K]
        """
        assert len(fea_s) == len(fea_t), "Student/teacher feature list length mismatch."

        b = fea_s[0].size(0)
        k = nebor_idx.size(1)
        total_loss = fea_s[0].new_tensor(0.0)

        for f_s, f_t in zip(fea_s, fea_t):
            if f_s.dim() != 4 or f_t.dim() != 4:
                continue

            _, c_s, h_s, w_s = f_s.shape
            _, c_t, h_t, w_t = f_t.shape

            # spatial alignment
            if h_s > h_t or w_s > w_t:
                f_s = F.adaptive_avg_pool2d(f_s, (h_t, w_t))
            elif h_s < h_t or w_s < w_t:
                f_t = F.adaptive_avg_pool2d(f_t, (h_s, w_s))

            _, _, h, w = f_s.shape
            hw = h * w

            # [B, C, HW]
            f_s = f_s.view(b, c_s, hw)
            f_t = f_t.view(b, c_t, hw)

            idx_flat = nebor_idx.reshape(-1)  # [B*K]

            # [B, K, C, HW]
            nebor_s_f = torch.index_select(f_s, 0, idx_flat).view(b, k, c_s, hw)
            nebor_t_f = torch.index_select(f_t, 0, idx_flat).view(b, k, c_t, hw)

            # center sample
            f_s_center = f_s.unsqueeze(1)  # [B, 1, C_s, HW]
            f_t_center = f_t.unsqueeze(1)  # [B, 1, C_t, HW]

            fea_s_diff = f_s_center - nebor_s_f
            fea_t_diff = f_t_center - nebor_t_f

            # [B, K, HW]
            fea_s_diff_spatial = F.normalize(
                fea_s_diff.pow(2).mean(dim=2), p=2, dim=2
            )
            fea_t_diff_spatial = F.normalize(
                fea_t_diff.pow(2).mean(dim=2), p=2, dim=2
            )

            loss_layer = torch.norm(
                fea_s_diff_spatial - fea_t_diff_spatial, p=2, dim=2
            ).pow(2).mean()

            total_loss += loss_layer

        return total_loss

    def res(self, logits_s, logits_t, nebor_idx):
        """
        Logits residual relation distillation.
        logits_s, logits_t: [B, C]
        nebor_idx: [B, K]
        """
        b = logits_s.size(0)
        k = nebor_idx.size(1)

        idx_flat = nebor_idx.reshape(-1)

        nebor_s_p = torch.index_select(logits_s, 0, idx_flat).view(b, k, -1)
        nebor_t_p = torch.index_select(logits_t, 0, idx_flat).view(b, k, -1)

        l_s = logits_s.unsqueeze(1)  # [B, 1, C]
        l_t = logits_t.unsqueeze(1)  # [B, 1, C]

        res_s_diff = l_s - nebor_s_p
        res_t_diff = l_t - nebor_t_p

        loss_nebor_res = self.js_div(res_s_diff, res_t_diff) / (b * k)
        return loss_nebor_res

    def nebor_loss(self, fea_s, fea_t, logits_s, logits_t):
        """
        Build neighborhood using teacher logits, then compute feature + residual losses.
        """
        if self.choose != "angle":
            raise ValueError("Unsupported choose='{}' for NKD".format(self.choose))

        _, idx = self.angle(logits_t)

        # first column is itself
        max_k = idx.size(1) - 1
        use_k = min(self.k, max_k)
        nebor_idx = idx[:, 1:use_k + 1]

        loss_fea = self.fea(fea_s, fea_t, nebor_idx)
        loss_res = self.res(logits_s, logits_t, nebor_idx)

        loss_nd = self.lambda2 * loss_fea + self.lambda1 * loss_res
        return loss_nd

    def forward(self, fea_s, fea_t, logits_s, logits_t):
        return self.nebor_loss(fea_s, fea_t, logits_s, logits_t)


class NKD(Distiller):
    """
    NKD distiller wrapper for the mdistiller framework.
    """

    def __init__(self, student, teacher, cfg):
        super(NKD, self).__init__(student, teacher)

        self.ce_loss_weight = cfg.NKD.CE_WEIGHT
        self.kd_loss_weight = cfg.NKD.KD_WEIGHT

        self.nkd_loss = NKDLoss(
            k=cfg.NKD.K,
            choose=cfg.NKD.CHOOSE,
            lambda1=cfg.NKD.LAMBDA1,
            lambda2=cfg.NKD.LAMBDA2,
        )

    def get_extra_parameters(self):
        return 0

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_dict_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_dict_teacher = self.teacher(image)

        if isinstance(feature_dict_student, dict):
            if "feats" in feature_dict_student:
                feat_student = feature_dict_student["feats"]
            else:
                raise KeyError(
                    "Student feature_dict does not contain key 'feats'. "
                    f"Available keys: {list(feature_dict_student.keys())}"
                )
        else:
            feat_student = feature_dict_student

        if isinstance(feature_dict_teacher, dict):
            if "feats" in feature_dict_teacher:
                feat_teacher = feature_dict_teacher["feats"]
            else:
                raise KeyError(
                    "Teacher feature_dict does not contain key 'feats'. "
                    f"Available keys: {list(feature_dict_teacher.keys())}"
                )
        else:
            feat_teacher = feature_dict_teacher

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_nkd = self.kd_loss_weight * self.nkd_loss(
            feat_student, feat_teacher, logits_student, logits_teacher
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_nkd,
        }
        return logits_student, losses_dict