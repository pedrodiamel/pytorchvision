import torch
from torch.optim.lr_scheduler import _LRScheduler


class CyclicLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        init_lr=1e-4,
        num_epochs_per_cycle=5,
        cycle_epochs_decay=2,
        lr_decay_factor=0.5,
        last_epoch=-1,
    ):
        self.init_lr = init_lr
        self.num_epochs_per_cycle = num_epochs_per_cycle
        self.cycle_epochs_decay = cycle_epochs_decay
        self.lr_decay_factor = lr_decay_factor
        self.last_epoch = last_epoch
        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch_in_cycle = self.last_epoch % self.num_epochs_per_cycle
        return [
            self.init_lr * (self.lr_decay_factor ** (epoch_in_cycle // self.cycle_epochs_decay))
            for base_lr in self.base_lrs
        ]
