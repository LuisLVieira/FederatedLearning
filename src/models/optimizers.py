from torch import optim

def get_optimizer_class():
    return {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "Adadelta": optim.Adadelta,
    }

def build_scheduler(optimizer, cfg):
    if "lr_scheduler" not in cfg or not cfg["lr_scheduler"]:
        return None

    name = next(iter(cfg["lr_scheduler"].keys()))
    params = cfg["lr_scheduler"][name]

    SchedulerClass = getattr(optim.lr_scheduler, name)

    scheduler = SchedulerClass(optimizer, **params)

    return scheduler