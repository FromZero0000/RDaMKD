from torchvision import transforms


class MultiViewTransform:
    """Apply independent transforms to one source image."""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [transform(image) for transform in self.transforms]


def rand_augment():
    if not hasattr(transforms, "RandAugment"):
        raise RuntimeError("MLKD requires torchvision with RandAugment support.")
    return transforms.RandAugment(num_ops=2, magnitude=10)
