import torch
import torch.nn as nn
import torch.nn.functional as F

#蒸馏器通用结构定义
class Distiller(nn.Module):
    #初始化
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
    #训练重定义
    #无论整个蒸馏器切到哪种模式，都强制 teacher 处在 eval 模式；而 student 跟随 mode。
    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
        #防御式编程：如果有人传了非布尔值（比如 1 / "train"），直接抛错，避免隐式转换导致的奇怪行为。
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self
    #获取可学习参数，只返回学生模型的所有参数
    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]
    #计算额外参数
    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0
    #前向训练
    #抽象接口，定义“训练模式”下该怎么前向。
    #类不实现，强制子类（如 DKD, CRD）去实现
    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()
    #前向测试过程
    def forward_test(self, image):
        return self.student(image)[0]
    #前向函数
    #参数都通过 **kwargs 传进来，方便不同蒸馏法自取所需键
    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
