import torch

from utils.transforms import *


class Config:
    def __init__(self, task_id=1):
        self.task_id = task_id
        self.output_dir = "output"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_device = torch.cuda.device_count()

        self.dataset = "ImageFolder"
        self.data_dir = "/data/face/parsing/dataset/CelebAMask-HQ_processed"
        self.image_size = 512
        self.crop_size = (448, 448)
        self.do_val = True

        self.train_transform = Compose([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                        RandomScale((0.75, 1.25)),
                                        RandomRotation(),
                                        RandomCrop(self.crop_size),
                                        ToTensor(),
                                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.val_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.model_name = "BiSeNet"
        self.model_args = Dict(
            backbone="STDCNet813",
            pretrain_model="snapshot/STDCNet813M_73.91.tar",
            use_boundary_2=False,
            use_boundary_4=False,
            use_boundary_8=True,
            use_boundary_16=False,
            use_conv_last=False
        )

        self.loss_name = "Loss"
        self.loss_args = Dict(score_thresh=0.7, ignore_idx=255)

        self.optimizer_name = "Optimizer"
        self.optimizer_args = Dict(
            momentum=0.9,
            weight_decay=5e-4,
            warmup_start_lr=1e-5,
            power=0.9,
        )

        self.lr = 0.01
        self.batch_size = 8
        self.milestones = Dict()
        self.epochs = 30

    def build(self, steps=None):
        if "lr0" not in self.optimizer_args:
            self.optimizer_args["lr0"] = self.lr

        if "max_iter" not in self.optimizer_args and steps is not None:
            self.optimizer_args["max_iter"] = self.epochs * steps

        if "warmup_steps" not in self.optimizer_args and steps is not None:
            self.optimizer_args["warmup_steps"] = steps

        return self


class Dict(dict):
    def __getattr__(self, item):
        return self.get(item, None)

    def __setattr__(self, key, value):
        self[key] = value
