import torch
import torch.nn.functional as F

from ._base import Distiller
from .DKD import dkd_loss
from .mlkd_compat import kd_loss, mlkd_losses


def _feature_list(output):
    if not isinstance(output, dict) or "feats" not in output:
        raise ValueError("Model output must contain a 'feats' dictionary entry.")
    features = output["feats"]
    if not isinstance(features, (list, tuple)) or not all(
        isinstance(feature, torch.Tensor) for feature in features
    ):
        raise TypeError("Model output 'feats' must be a sequence of tensors.")
    features = list(features)
    if isinstance(output.get("pooled_feat"), torch.Tensor):
        features.append(output["pooled_feat"])
    if not features:
        raise ValueError("Model output contains no usable feature tensors.")
    return features


def _resolve_layer_pairs(student_count, teacher_count, layer_ids):
    if student_count <= 0 or teacher_count <= 0:
        raise ValueError("Teacher and student must both provide feature tensors.")
    if not layer_ids:
        raise ValueError("LAYER_IDS must contain at least one layer index.")

    pairs = []
    for layer_id in layer_ids:
        student_id = layer_id if layer_id >= 0 else student_count + layer_id
        teacher_id = layer_id if layer_id >= 0 else teacher_count + layer_id
        if not 0 <= student_id < student_count:
            raise IndexError(
                "Student layer index {} is out of range.".format(layer_id)
            )
        if not 0 <= teacher_id < teacher_count:
            raise IndexError(
                "Teacher layer index {} is out of range.".format(layer_id)
            )
        pair = (student_id, teacher_id)
        if pair in pairs:
            raise ValueError("LAYER_IDS contains a duplicate layer index.")
        pairs.append(pair)
    return pairs


def _normalize_layer_weights(layer_count, weights, device):
    if len(weights) != layer_count:
        raise ValueError("LAYER_WEIGHTS must match the number of selected layers.")
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("LAYER_WEIGHTS must be finite and non-negative.")
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("LAYER_WEIGHTS must have a positive sum.")
    return weights / total_weight


def _gap(feature):
    """Normalize an arbitrary feature tensor to [batch, channels]."""
    if feature.dim() == 4:
        return F.adaptive_avg_pool2d(feature, 1).flatten(1)
    if feature.dim() == 3:
        return feature.mean(dim=-1)
    if feature.dim() == 2:
        return feature
    return feature.flatten(1)


def _pad_feature_dims(teacher_feature, student_feature):
    """Zero-pad the lower-dimensional representation."""
    common_dim = max(teacher_feature.size(1), student_feature.size(1))
    teacher_feature = F.pad(
        teacher_feature, (0, common_dim - teacher_feature.size(1))
    )
    student_feature = F.pad(
        student_feature, (0, common_dim - student_feature.size(1))
    )
    return teacher_feature, student_feature


def _class_prototypes(teacher_feature, student_feature, labels):
    teacher_prototypes, student_prototypes = [], []
    for class_id in torch.unique(labels.detach(), sorted=True):
        class_mask = labels == class_id
        teacher_prototypes.append(teacher_feature[class_mask].mean(0))
        student_prototypes.append(student_feature[class_mask].mean(0))
    if len(teacher_prototypes) <= 1:
        return None
    return torch.stack(teacher_prototypes), torch.stack(student_prototypes)


@torch.no_grad()
def _fit_subspace(feature, subspace_dim):
    """Fit an orthonormal PCA basis on the feature dimension."""
    with torch.cuda.amp.autocast(enabled=False):
        feature = feature.float()
        feature = feature - feature.mean(0, keepdim=True)
        effective_dim = max(
            1,
            min(
                int(subspace_dim),
                feature.size(1),
                feature.size(0) - 1,
            ),
        )
        _, _, right_vectors = torch.linalg.svd(feature, full_matrices=False)
        basis = right_vectors.transpose(-1, -2)[:, :effective_dim]
        return torch.linalg.qr(basis, mode="reduced").Q

def _pairwise_cosine(feature):
    feature = F.normalize(feature, p=2, dim=1, eps=1.0e-8)
    return feature @ feature.t()

@torch.no_grad()
def _gfk_kernel(teacher_basis: torch.Tensor, student_basis: torch.Tensor):
    """Construct the shared manifold-aware representation-domain operator."""
    #The core code will be available after acceptance
    raise NotImplementedError("Implementation omitted from this version.")

def _geometry_loss(teacher_feature, student_feature, topk=5):
    """Match the teacher and student top-k cosine neighborhoods."""
    with torch.no_grad():
        teacher_cosine = _pairwise_cosine(teacher_feature)
        batch_size = teacher_cosine.size(0)
        if batch_size <= 1:
            return teacher_feature.new_tensor(0.0)
        teacher_cosine.fill_diagonal_(
            torch.finfo(teacher_cosine.dtype).min
        )
        neighbor_count = min(max(1, int(topk)), batch_size - 1)
        neighbors = teacher_cosine.topk(neighbor_count, dim=1).indices
        mask = torch.zeros_like(teacher_cosine, dtype=torch.bool)
        mask.scatter_(1, neighbors, True)
    difference = (
        teacher_cosine - _pairwise_cosine(student_feature)
    )[mask]
    return difference.square().mean()

def _mmd_linear(teacher_feature, student_feature):
    mean_difference = teacher_feature.mean(0) - student_feature.mean(0)
    return mean_difference.square().sum()

def _mmd_rbf(teacher_feature, student_feature, sigma=1.0):
    scale = 2.0 * max(float(sigma), 1.0e-12) ** 2

    def kernel_mean(left, right):
        squared_distance = torch.cdist(left, right).square()
        return torch.exp(-squared_distance / scale).mean()

    return (
        kernel_mean(teacher_feature, teacher_feature)
        + kernel_mean(student_feature, student_feature)
        - 2.0 * kernel_mean(teacher_feature, student_feature)
    )

def _mmd_loss(teacher_feature, student_feature, kernel, sigma):
    if kernel.lower() == "linear":
        return _mmd_linear(teacher_feature, student_feature)
    if kernel.lower() == "rbf":
        return _mmd_rbf(teacher_feature, student_feature, sigma)
    raise ValueError("MMD kernel must be 'linear' or 'rbf'.")


def _linear_cka(teacher_feature, student_feature, eps=1.0e-6):
    """Compute linear centered kernel alignment."""
    teacher_feature = teacher_feature - teacher_feature.mean(
        0, keepdim=True
    )
    student_feature = student_feature - student_feature.mean(
        0, keepdim=True
    )
    teacher_gram = teacher_feature @ teacher_feature.t()
    student_gram = student_feature @ student_feature.t()
    numerator = (teacher_gram * student_gram).sum()
    denominator = (
        torch.linalg.norm(teacher_gram, ord="fro")
        * torch.linalg.norm(student_gram, ord="fro")
    )
    return numerator / (denominator + eps)


class ManifoldAlignKD_CKA(Distiller):
    """RDaMKD with a selectable KD, DKD, or MLKD logit objective."""

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        method_cfg = getattr(cfg, "ManifoldAlignKD_CKA", None)

        def get(node, key, default):
            return getattr(node, key, default) if node is not None else default

        self.ce_weight = float(get(method_cfg, "CE_WEIGHT", 1.0))
        self.kd_weight = float(get(method_cfg, "KD_WEIGHT", 1.0))
        self.temperature = float(get(method_cfg, "KD_T", 4.0))
        self.dkd = cfg.DKD
        self.dkd_warmup = max(1.0, float(get(self.dkd, "WARMUP", 20)))
        self.align_warmup = max(
            1.0, float(get(method_cfg, "WARMUP", 1))
        )

        self.mlkd_temperature = float(cfg.KD.TEMPERATURE)
        self.mlkd_ce_weight = float(cfg.KD.LOSS.CE_WEIGHT)
        self.mlkd_kd_weight = float(cfg.KD.LOSS.KD_WEIGHT)
        self.logit_stand = bool(get(cfg.EXPERIMENT, "LOGIT_STAND", False))

        self.layer_ids = list(get(method_cfg, "LAYER_IDS", [-1]))
        self.layer_weights = list(get(method_cfg, "LAYER_WEIGHTS", []))
        self.subspace_dim = int(get(method_cfg, "SUBSPACE_DIM", 32))

        mmd_cfg = get(method_cfg, "MMD", None)
        self.mmd_kernel = str(get(mmd_cfg, "KERNEL", "linear"))
        self.mmd_sigma = float(get(mmd_cfg, "SIGMA", 1.0))
        self.lambda_mmd = float(get(method_cfg, "LAMBDA_MMD", 1.0))
        self.lambda_geo = float(get(method_cfg, "LAMBDA_GEO", 1.0))
        self.topk = int(get(method_cfg, "GEO_TOPK", 8))

        mu_cfg = get(method_cfg, "MU", None)
        self.mu_mode = str(get(mu_cfg, "MODE", "cka"))
        self.mu_base = float(get(mu_cfg, "BASE", 0.5))
        self.mu_alpha = float(get(mu_cfg, "ALPHA", 3.0))
        self.mu_center = float(get(mu_cfg, "CENTER", 0.5))

        self.base_type = str(get(method_cfg, "BASE_TYPE", "dkd")).lower()
        if self.base_type not in {"none", "kd", "dkd", "mlkd"}:
            raise ValueError("BASE_TYPE must be one of: none, kd, dkd, mlkd.")
        self.use_class_prototypes = bool(
            get(method_cfg, "CLASS_PROTOTYPE", False)
        )

    def _estimate_mu(self, teacher_features, student_features):
        """Estimate the adaptive balance between global and local alignment."""
        raise NotImplementedError("Implementation omitted from this version.")

    def _base_losses(
        self,
        student_weak,
        teacher_weak,
        target,
        epoch,
        student_strong=None,
        teacher_strong=None,
    ):
        if self.base_type == "mlkd":
            if student_strong is None or teacher_strong is None:
                raise ValueError(
                    "BASE_TYPE='mlkd' requires weak and strong image views."
                )
            losses = mlkd_losses(
                student_weak,
                student_strong,
                teacher_weak,
                teacher_strong,
                target,
                self.mlkd_temperature,
                self.mlkd_ce_weight,
                self.mlkd_kd_weight,
                self.logit_stand,
            )
            loss_ce = losses.pop("loss_ce")
            return loss_ce, sum(losses.values())

        loss_ce = self.ce_weight * F.cross_entropy(student_weak, target)
        if self.base_type == "kd":
            loss_base = self.kd_weight * kd_loss(
                student_weak, teacher_weak, self.temperature
            )
        elif self.base_type == "dkd":
            loss_base = min(epoch / self.dkd_warmup, 1.0) * dkd_loss(
                student_weak,
                teacher_weak,
                target,
                self.dkd.ALPHA,
                self.dkd.BETA,
                self.dkd.T,
            )
        else:
            loss_base = student_weak.new_tensor(0.0)
        return loss_ce, loss_base

    def forward_train(
        self,
        image=None,
        target=None,
        image_weak=None,
        image_strong=None,
        **kwargs,
    ):
        weak_image = image_weak if image_weak is not None else image
        if weak_image is None:
            raise ValueError("A training image or image_weak tensor is required.")

        student_logits, student_output = self.student(weak_image)
        teacher_image = kwargs.get("teacher_image")
        if teacher_image is None:
            teacher_image = weak_image
        with torch.no_grad():
            teacher_logits, teacher_output = self.teacher(teacher_image)

        student_strong_logits = teacher_strong_logits = None
        if self.base_type == "mlkd":
            if image_strong is None:
                raise ValueError(
                    "BASE_TYPE='mlkd' requires a strong augmented image view."
                )
            student_strong_logits, _ = self.student(image_strong)
            with torch.no_grad():
                teacher_strong_logits, _ = self.teacher(image_strong)

        student_features = _feature_list(student_output)
        teacher_features = _feature_list(teacher_output)
        layer_pairs = _resolve_layer_pairs(
            len(student_features), len(teacher_features), self.layer_ids
        )
        layer_weights = _normalize_layer_weights(
            len(layer_pairs), self.layer_weights, weak_image.device
        )

        loss_mmd = weak_image.new_tensor(0.0)
        loss_geo = weak_image.new_tensor(0.0)
        teacher_aligned, student_aligned = [], []

        for weight, (student_id, teacher_id) in zip(layer_weights, layer_pairs):
            teacher_feature, student_feature = _pad_feature_dims(
                _gap(teacher_features[teacher_id]),
                _gap(student_features[student_id]),
            )
            if teacher_feature.size(1) <= 1:
                raise ValueError(
                    "Selected features must contain more than one channel."
                )

            if self.use_class_prototypes:
                prototypes = _class_prototypes(
                    teacher_feature, student_feature, target
                )
                if prototypes is None:
                    raise ValueError(
                        "Class prototypes require at least two classes per batch."
                    )
                teacher_feature, student_feature = prototypes

            basis_dim = min(self.subspace_dim, teacher_feature.size(1) - 1)
            teacher_basis = _fit_subspace(teacher_feature, basis_dim)
            student_basis = _fit_subspace(student_feature, basis_dim)
            shared_dim = min(teacher_basis.size(1), student_basis.size(1))
            shared_operator = _gfk_kernel(
                teacher_basis[:, :shared_dim],
                student_basis[:, :shared_dim],
            )
            teacher_shared = teacher_feature @ shared_operator
            student_shared = student_feature @ shared_operator

            teacher_aligned.append(teacher_shared)
            student_aligned.append(student_shared)
            loss_mmd = loss_mmd + weight * _mmd_loss(
                teacher_shared,
                student_shared,
                self.mmd_kernel,
                self.mmd_sigma,
            )
            loss_geo = loss_geo + weight * _geometry_loss(
                teacher_shared, student_shared, self.topk
            )

        teacher_aligned = torch.cat(teacher_aligned, dim=1)
        student_aligned = torch.cat(student_aligned, dim=1)

        epoch = float(kwargs.get("epoch", 0))
        warmup = min(epoch / self.align_warmup, 1.0)
        mu, cka = self._estimate_mu(
            teacher_aligned.detach(), student_aligned.detach()
        )
        loss_align = warmup * (
            mu * self.lambda_mmd * loss_mmd
            + (1.0 - mu) * self.lambda_geo * loss_geo
        )

        loss_ce, loss_base = self._base_losses(
            student_logits,
            teacher_logits,
            target,
            epoch,
            student_strong_logits,
            teacher_strong_logits,
        )
        return student_logits, {
            "loss_ce": loss_ce,
            "loss_kd": loss_base + loss_align,
            "mu": mu.detach(),
            "cka": cka.detach(),
        }

    def get_learnable_parameters(self):
        return list(self.student.parameters())

    def get_extra_parameters(self):
        return 0
