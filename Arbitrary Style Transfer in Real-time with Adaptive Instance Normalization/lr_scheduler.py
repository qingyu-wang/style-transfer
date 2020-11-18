class LRScheduler(object):

    def __init__(self, optimizer, lr_begin, lr_decay):
        self.optimizer = optimizer
        self.lr_begin = lr_begin
        self.lr_decay = lr_decay

    def __call__(self, iter_cnt):
        self.lr = self.lr_begin / (1.0 + self.lr_decay * iter_cnt)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
        return
