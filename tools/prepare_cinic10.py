import argparse
import os
import random
import tarfile
import time

import requests


CINIC10_URL = (
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/"
    "CINIC-10.tar.gz?sequence=4&isAllowed=y"
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def repo_data_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print("Using existing archive: {}".format(dst))
        return
    tmp = dst + ".part"
    headers = {}
    mode = "wb"
    existing = 0
    if os.path.isfile(tmp):
        existing = os.path.getsize(tmp)
        if existing > 0:
            headers["Range"] = "bytes={}-".format(existing)
            mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=30) as response:
        response.raise_for_status()
        total = response.headers.get("content-length")
        total = int(total) + existing if total is not None else None
        seen = existing
        last = time.time()
        with open(tmp, mode) as writer:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                writer.write(chunk)
                seen += len(chunk)
                now = time.time()
                if now - last > 5:
                    if total:
                        print("Downloaded {:.1f}/{:.1f} MB".format(seen / 2**20, total / 2**20))
                    else:
                        print("Downloaded {:.1f} MB".format(seen / 2**20))
                    last = now
    os.replace(tmp, dst)
    print("Downloaded archive: {}".format(dst))


def extract(archive, root):
    marker = os.path.join(root, "train")
    if os.path.isdir(marker):
        print("Using existing extracted CINIC-10: {}".format(root))
        return
    data_root = os.path.dirname(root)
    os.makedirs(data_root, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_root)
    if not os.path.isdir(marker):
        os.makedirs(root, exist_ok=True)
        for name in [
            "train",
            "valid",
            "test",
            "README.md",
            "imagenet-contributors.csv",
            "synsets-to-cifar-10-classes.txt",
        ]:
            src = os.path.join(data_root, name)
            if os.path.exists(src):
                os.replace(src, os.path.join(root, name))
    print("Extracted CINIC-10 to: {}".format(root))


def is_cifar_origin(filename):
    name = os.path.basename(filename).lower()
    return name.startswith("cifar10")


def build_manifests(cinic_root, manifest_root):
    os.makedirs(manifest_root, exist_ok=True)
    for split in ["train", "valid", "test"]:
        lines = []
        raw_count = 0
        removed_count = 0
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(cinic_root, split, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError("Missing CINIC-10 class folder: {}".format(class_dir))
            for filename in sorted(os.listdir(class_dir)):
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                raw_count += 1
                if is_cifar_origin(filename):
                    removed_count += 1
                    continue
                rel_path = os.path.join(split, class_name, filename).replace("\\", "/")
                lines.append("{}\t{}".format(rel_path, label))

        manifest = os.path.join(manifest_root, "{}.txt".format(split))
        with open(manifest, "w") as writer:
            writer.write("\n".join(lines))
            writer.write("\n")
        print(
            "{}: kept {}, removed CIFAR-origin {}, raw {}".format(
                split, len(lines), removed_count, raw_count
            )
        )


def build_cifar_size_manifests(clean_manifest_root, out_manifest_root, seed=10):
    split_plan = {
        "train": 5000,
        "valid": 1000,
        "test": 1000,
    }
    os.makedirs(out_manifest_root, exist_ok=True)
    rng = random.Random(seed)

    for split, per_class in split_plan.items():
        src_manifest = os.path.join(clean_manifest_root, "{}.txt".format(split))
        if not os.path.isfile(src_manifest):
            raise FileNotFoundError(
                "Missing clean manifest: {}. Run "
                "`python tools/prepare_cinic10.py --manifest` first.".format(src_manifest)
            )

        by_class = {label: [] for label in range(len(CLASS_NAMES))}
        with open(src_manifest, "r") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.split("\t")
                by_class[int(label)].append((rel_path, int(label)))

        selected = []
        for label in range(len(CLASS_NAMES)):
            samples = by_class[label]
            if len(samples) < per_class:
                raise ValueError(
                    "{} class {} has only {} samples, need {}".format(
                        split, label, len(samples), per_class
                    )
                )
            samples = list(samples)
            rng.shuffle(samples)
            selected.extend(samples[:per_class])

        selected.sort(key=lambda item: (item[1], item[0]))
        dst_manifest = os.path.join(out_manifest_root, "{}.txt".format(split))
        with open(dst_manifest, "w") as writer:
            for rel_path, label in selected:
                writer.write("{}\t{}\n".format(rel_path, label))
        print(
            "{}: wrote {} balanced samples to {}".format(
                split, len(selected), dst_manifest
            )
        )


def main():
    parser = argparse.ArgumentParser("Prepare cleaned CINIC-10 manifests.")
    parser.add_argument("--data-root", default=repo_data_dir())
    parser.add_argument("--url", default=CINIC10_URL)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--cifar-size-manifest", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    archive = os.path.join(args.data_root, "CINIC-10.tar.gz")
    cinic_root = os.path.join(args.data_root, "cinic-10")
    manifest_root = os.path.join(args.data_root, "cinic-10-clean")
    cifar_size_manifest_root = os.path.join(args.data_root, "cinic-10-clean-cifar-size")

    if args.download:
        download(args.url, archive)
    if args.extract:
        extract(archive, cinic_root)
    if args.manifest:
        build_manifests(cinic_root, manifest_root)
    if args.cifar_size_manifest:
        build_cifar_size_manifests(
            clean_manifest_root=manifest_root,
            out_manifest_root=cifar_size_manifest_root,
            seed=args.seed,
        )
    if not (args.download or args.extract or args.manifest or args.cifar_size_manifest):
        parser.print_help()


if __name__ == "__main__":
    main()
