from .cifar10 import get_cifar10_dataloaders
from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .cinic10 import (
    get_cinic10_clean_cifar_size_dataloaders,
    get_cinic10_clean_cifar_size_paired_dataloaders,
    get_cinic10_clean_dataloaders,
)
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample
from .tiny_imagenet import get_tinyimagenet_dataloader, get_tinyimagenet_dataloader_sample


def _uses_mlkd(cfg):
    distiller_type = str(cfg.DISTILLER.TYPE).lower()
    if distiller_type == "mlkd":
        return True
    method_cfg = getattr(cfg, "ManifoldAlignKD_CKA", None)
    return (
        distiller_type == "manifoldalignkd_cka"
        and str(getattr(method_cfg, "BASE_TYPE", "")).lower() == "mlkd"
    )


def get_dataset(cfg):
    strong = _uses_mlkd(cfg)
    if strong and cfg.DATASET.TYPE not in {"cifar100", "imagenet"}:
        raise NotImplementedError(
            "MLKD weak/strong views are currently available for CIFAR-100 "
            "and ImageNet."
        )

    if cfg.DATASET.TYPE == "cifar10":
        train_loader, val_loader, num_data = get_cifar10_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        num_classes = 10
    elif cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                strong=strong,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                strong=strong,
            )
        num_classes = 1000
    elif cfg.DATASET.TYPE == "tiny_imagenet":
        if cfg.DISTILLER.TYPE in ("CRD", "CRDKD"):
            train_loader, val_loader, num_data = get_tinyimagenet_dataloader_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_tinyimagenet_dataloader(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 200
    elif cfg.DATASET.TYPE == "cinic10_clean":
        train_loader, val_loader, num_data = get_cinic10_clean_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        num_classes = 10
    elif cfg.DATASET.TYPE == "cinic10_clean_cifar_size":
        train_loader, val_loader, num_data = get_cinic10_clean_cifar_size_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        num_classes = 10
    elif cfg.DATASET.TYPE == "cinic10_clean_cifar_size_paired":
        train_loader, val_loader, num_data = (
            get_cinic10_clean_cifar_size_paired_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        )
        num_classes = 10
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes
