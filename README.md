# RDaMKD

PyTorch implementation of **Bridging Teacher–Student Representation Domains for Manifold-Aware Knowledge Distillation**.

RDaMKD constructs a shared manifold-aware domain between teacher and student representations, then performs global distribution alignment and teacher-guided local geometry preservation in that domain.

This repository is developed on top of [MDistiller](https://github.com/megvii-research/mdistiller), the official codebase of [Decoupled Knowledge Distillation (DKD)](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html). We thank the DKD authors for releasing their code.

## Environment

### Reference setup

- Ubuntu 20.04/22.04
- Python 3.8
- PyTorch 1.9.0
- torchvision 0.10.0
- CUDA 11.1 with a compatible NVIDIA driver

The current training and evaluation entry points move models and data to CUDA
directly, so an NVIDIA GPU is required.

Create the reference environment:

```bash
conda create -n rdamkd python=3.8 -y
conda activate rdamkd

conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1 \
  -c pytorch -c conda-forge

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

If a different CUDA version is installed, use the matching PyTorch and
torchvision packages from the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).
Weights & Biases integration is retained as an optional compatibility feature,
but it is disabled in the default configuration and in all provided experiment
files. The released code records local worklogs and TensorBoard events.

Confirm the CUDA environment before running an experiment:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available()"
```

Run all commands below from the repository root.

## Evaluate

The provided CIFAR-100 student checkpoints are stored in `output/ckpt/`.
Checkpoint filenames follow the teacher–student naming convention, while
`--model` specifies the student architecture. For example:

```bash
python tools/eval.py \
  --dataset cifar100 \
  --model resnet8x4 \
  --ckpt output/ckpt/resnet32x4_resnet8x4.ckpt \
  --batch-size 64
```

The script downloads CIFAR-100 to `./data` when it is first used and reports
Top-1 accuracy, Top-5 accuracy, and cross-entropy loss. To evaluate another
provided checkpoint, change the checkpoint path and student model name in the
same command.

## Train

### CIFAR-100

Download the pretrained teacher checkpoints from the
[MDistiller checkpoint release](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints)
and extract them under `download_ckpts/cifar_teachers/`, following the original
MDistiller directory layout. For example, the ResNet32x4 teacher checkpoint is
expected at:

```text
download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth
```

Start an RDaMKD experiment with:

```bash
python tools/train.py \
  --cfg configs/cifar100/manifoldkd_cka/res32x4_res8x4.yaml
```

Additional teacher–student configurations are available in
`configs/cifar100/manifoldkd_cka/`. CIFAR-100 is downloaded automatically.
RDaMKD worklogs are written to `output/log/` using the
`<teacher>_<student>_worklog.txt` naming convention. Training checkpoints and
TensorBoard events remain under the experiment directory below the configured
output prefix.

### ImageNet

```bash
python tools/train.py \
  --cfg configs/imagenet/r34_r18/manifold_cka_kd.yaml
```

### Logit distillation objective

`ManifoldAlignKD_CKA.BASE_TYPE` supports `none`, `kd`, `dkd`, and `mlkd`.
Select the objective in the YAML file or override it from the command line:

```bash
python tools/train.py \
  --cfg configs/cifar100/manifoldkd_cka/res32x4_res8x4.yaml \
  ManifoldAlignKD_CKA.BASE_TYPE mlkd
```

The MLKD option automatically enables the weak/strong augmented image views
required by its multi-level logit objective.

## License

Original RDaMKD code and MIT-licensed MDistiller-derived portions are provided
under the [MIT License](LICENSE). Third-party notices are included in the same
license file.
