from .HardNegativeNLLLoss import HardNegativeNLLLoss
from .DPOLoss import DPOLoss

def load_loss(loss_class, *args, **kwargs):
    if loss_class == "HardNegativeNLLLoss":
        loss_cls = HardNegativeNLLLoss
    elif loss_class == "DPOLoss":
        loss_cls = DPOLoss
    else:
        raise ValueError(f"Unknown loss class {loss_class}")
    return loss_cls(*args, **kwargs)
