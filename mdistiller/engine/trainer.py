import os
import time
from tqdm import tqdm
import torch
import torch.optim as optim
from collections import OrderedDict
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from .dot import DistillationOrientedTrainer
from torch.optim.lr_scheduler import CosineAnnealingLR


def _move_images_to_cuda(image):
    if isinstance(image, (list, tuple)):
        if len(image) != 2:
            raise ValueError("Expected exactly two image views: weak and strong.")
        return tuple(
            view.float().cuda(non_blocking=True) for view in image
        )
    return image.float().cuda(non_blocking=True)


#基础训练器
class BaseTrainer(object):
    #初始化
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1
        self.base_lr = cfg.SOLVER.LR
        self.warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
        self.warmup_start_lr = cfg.SOLVER.WARMUP_START_LR
        self.LrMode = cfg.SOLVER.LR_MODE
        self.best_epoch = 0
        if self.LrMode == 'cos':
            t_max = max(1, cfg.SOLVER.EPOCHS - self.warmup_epochs)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=cfg.SOLVER.ETA_MIN)

        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.worklog_path = os.path.join(self.log_path, "worklog.txt")
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    #初始化优化器
    def init_optimizer(self, cfg):
        #如果采用SGD训练器
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g["lr"] = lr
    def _get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    #加载日志
    def log(self, lr, epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb
            wandb.log({"current lr": lr})
            wandb.log(log_dict)

        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
                wandb.run.summary["best_epoch"] = self.best_epoch
        # worklog.txt
        with open(self.worklog_path, "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.8g}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            writer.writelines(lines)

    def train(self, resume=False):
        # epoch置1
        epoch = 1
        # 如果有断点，则加载断点处信息
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        # 如果训练epoch内，循环训练
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            #一个epoch的训练函数
            self.train_epoch(epoch)
            epoch += 1
        #训练完打印最高识别率
        print(log_msg("Best accuracy:{},Best epoch:{}".format(self.best_acc, self.best_epoch), "EVAL"))
        #存入worklog文档中
        with open(self.worklog_path, "a") as writer:
            writer.write(
                "-" * 25
                + os.linesep
                + "best_acc: "
                + "{:.2f}".format(float(self.best_acc))
                + os.linesep
            )

    # 1个epoch的训练函数
    def train_epoch(self, epoch):
        # === 学习率调度（支持 step 与 cosine+warmup）===
        if self.LrMode == "cos":
            # warmup：前 warmup_epochs 线性从 warmup_start_lr -> base_lr
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                ratio = epoch / float(self.warmup_epochs)
                warm_lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * ratio
                self._set_lr(warm_lr)
            else:
                # cos衰减
                self.scheduler.step()
            lr = self._get_lr()
        else:
            # step衰减
            lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        print(f"[LR] epoch={epoch} lr={lr:.8f}")
        #训练过程中的临时参数变量
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        #训练实例数量
        num_iter = len(self.train_loader)
        #进度条
        pbar = tqdm(range(num_iter))
        #设置为训练模式
        self.distiller.train()
        #循环训练
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        #关闭进度条
        pbar.close()
        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
            }
        )
        self.log(lr, epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,

        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )
            self.best_epoch = epoch



    #一个实例训练函数
    def train_iter(self, data, epoch, train_meters):
        #梯度清零
        self.optimizer.zero_grad()
        train_start_time = time.time()
        #读取实例数据
        teacher_image = None
        if len(data) == 4:
            image, target, index, teacher_image = data
        else:
            image, target, index = data
        #更新读取时间
        train_meters["data_time"].update(time.time() - train_start_time)
        #将图片转移到GPU上
        image = _move_images_to_cuda(image)
        if teacher_image is not None:
            teacher_image = teacher_image.float()
            teacher_image = teacher_image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward过程
        if isinstance(image, tuple):
            image_weak, image_strong = image
            preds, losses_dict = self.distiller(
                image_weak=image_weak,
                image_strong=image_strong,
                target=target,
                epoch=epoch,
            )
            batch_size = image_weak.size(0)
        else:
            preds, losses_dict = self.distiller(
                image=image, target=target, epoch=epoch, teacher_image=teacher_image
            )
            batch_size = image.size(0)

        # backward过程
        # Diagnostic values such as mu and CKA are not optimization terms.
        loss_dict = {
            key: value
            for key, value in losses_dict.items()
            if key.startswith("loss_")
        }
        optimization_terms = (
            loss_dict.values() if loss_dict else losses_dict.values()
        )
        loss = sum(value.mean() for value in optimization_terms)
        loss.backward()
        # 更新参数
        self.optimizer.step()
        # 更新训练时间
        train_meters["training_time"].update(time.time() - train_start_time)
        # 更新实验结果与信息
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # 输出信息
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = _move_images_to_cuda(image)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class DOT(BaseTrainer):
    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(self.worklog_path, "a") as writer:
            writer.write(
                "-" * 25
                + os.linesep
                + "best_acc: "
                + "{:.2f}".format(float(self.best_acc))
                + os.linesep
            )

    def train_iter(self, data, epoch, train_meters):
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDDOT(BaseTrainer):

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(self.worklog_path, "a") as writer:
            writer.write(
                "-" * 25
                + os.linesep
                + "best_acc: "
                + "{:.2f}".format(float(self.best_acc))
                + os.linesep
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        # self.optimizer.step((1 - epoch / 240.))
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class MuLoggerTrainer(BaseTrainer):
    """
    专门给会返回 mu / cka / mmd / geo / align 的 distiller 使用。
    不改原 BaseTrainer，避免影响其他实验。
    """

    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        super().__init__(experiment_name, distiller, train_loader, val_loader, cfg)
        public_log_dir = os.path.join(cfg.LOG.PREFIX, "log")
        os.makedirs(public_log_dir, exist_ok=True)
        model_pair = "{}_{}".format(
            cfg.DISTILLER.TEACHER.lower(),
            cfg.DISTILLER.STUDENT.lower(),
        )
        self.worklog_path = os.path.join(
            public_log_dir, model_pair + "_worklog.txt"
        )

    def train_epoch(self, epoch):
        # ===== 学习率调度（与 BaseTrainer 保持一致）=====
        if self.LrMode == "cos":
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                ratio = epoch / float(self.warmup_epochs)
                warm_lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * ratio
                self._set_lr(warm_lr)
            else:
                self.scheduler.step()
            lr = self._get_lr()
        else:
            lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)

        print(f"[LR] epoch={epoch} lr={lr:.8f}")

        # ===== 新增 mu 等统计量 =====
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
            "mu": AverageMeter(),
            "cka": AverageMeter(),
        }

        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))
        self.distiller.train()

        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters, idx)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()

        pbar.close()
        # ===== validate =====
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

        # ===== epoch 级日志 =====
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "train_mu": train_meters["mu"].avg,
                "train_cka": train_meters["cka"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
            }
        )
        self.log(lr, epoch, log_dict)

        # ===== checkpoint 保存逻辑完全复用 =====
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}

        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(student_state, os.path.join(self.log_path, "student_latest"))

        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(state, os.path.join(self.log_path, "epoch_{}".format(epoch)))
            save_checkpoint(student_state, os.path.join(self.log_path, "student_{}".format(epoch)))

        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(student_state, os.path.join(self.log_path, "student_best"))
            self.best_epoch = epoch

    def train_iter(self, data, epoch, train_meters, batch_idx=0):
        self.optimizer.zero_grad()
        train_start_time = time.time()

        teacher_image = None
        if len(data) == 4:
            image, target, index, teacher_image = data
        else:
            image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)

        image = _move_images_to_cuda(image)
        if teacher_image is not None:
            teacher_image = teacher_image.float()
            teacher_image = teacher_image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # ===== forward =====
        if isinstance(image, tuple):
            image_weak, image_strong = image
            preds, losses_dict = self.distiller(
                image_weak=image_weak,
                image_strong=image_strong,
                target=target,
                epoch=epoch,
            )
            batch_size = image_weak.size(0)
        else:
            preds, losses_dict = self.distiller(
                image=image, target=target, epoch=epoch, teacher_image=teacher_image
            )
            batch_size = image.size(0)

        # ===== 只对 loss_* 做反向传播 =====
        loss_dict = {k: v for k, v in losses_dict.items() if k.startswith("loss_")}
        if len(loss_dict) == 0:
            raise ValueError("No keys starting with 'loss_' were found in losses_dict.")

        loss = sum([v.mean() for v in loss_dict.values()])

        # ===== backward =====
        loss.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)

        # ===== 基础统计 =====
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))

        train_meters["losses"].update(loss.detach().item(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)

        # ===== 额外统计：mu / cka / mmd / geo / align =====
        extra_keys = ["mu", "cka"]
        for k in extra_keys:
            if k in losses_dict:
                train_meters[k].update(losses_dict[k].mean().detach().item(), batch_size)

        # ===== optional batch-level WandB logging (disabled by default) =====
        if self.cfg.LOG.WANDB:
            iter_log = {}
            for k in extra_keys:
                if k in losses_dict:
                    iter_log[f"iter_{k}"] = losses_dict[k].mean().detach().item()
            if len(iter_log) > 0:
                import wandb
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                wandb.log(iter_log, step=global_step)

        msg = (
            "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| "
            "Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}| mu:{:.4f}| cka:{:.4f}"
        ).format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
            train_meters["mu"].avg,
            train_meters["cka"].avg,
        )
        return msg
