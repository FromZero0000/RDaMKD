"""Compatibility helpers around the unmodified official MLKD implementation."""

import sys
import types


def _provide_legacy_imports():
    """Provide unused legacy imports when they are unavailable."""
    try:
        from termios import CEOL  # noqa: F401
    except ImportError:
        termios = types.ModuleType("termios")
        termios.CEOL = None
        sys.modules.setdefault("termios", termios)

    try:
        from turtle import st  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        turtle = sys.modules.get("turtle", types.ModuleType("turtle"))
        turtle.st = None
        sys.modules["turtle"] = turtle


_provide_legacy_imports()

from .MLKD import MLKD, kd_loss  # noqa: E402


class _FixedLogits:
    def __init__(self, weak_key, strong_key, weak_logits, strong_logits):
        self.weak_key = weak_key
        self.strong_key = strong_key
        self.weak_logits = weak_logits
        self.strong_logits = strong_logits

    def __call__(self, image):
        if image is self.weak_key:
            return self.weak_logits, {}
        if image is self.strong_key:
            return self.strong_logits, {}
        raise ValueError("Unknown MLKD compatibility input.")


def mlkd_losses(
    student_weak,
    student_strong,
    teacher_weak,
    teacher_strong,
    target,
    temperature=4.0,
    ce_weight=1.0,
    kd_weight=1.0,
    logit_stand=False,
):
    """Run the official MLKD loss assembly on precomputed logits."""
    weak_key = object()
    strong_key = object()
    context = types.SimpleNamespace(
        student=_FixedLogits(
            weak_key,
            strong_key,
            student_weak,
            student_strong,
        ),
        teacher=_FixedLogits(
            weak_key,
            strong_key,
            teacher_weak,
            teacher_strong,
        ),
        temperature=temperature,
        ce_loss_weight=ce_weight,
        kd_loss_weight=kd_weight,
        logit_stand=logit_stand,
    )
    _, losses = MLKD.forward_train(
        context,
        image_weak=weak_key,
        image_strong=strong_key,
        target=target,
    )
    return losses
