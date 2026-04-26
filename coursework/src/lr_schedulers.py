# Copyright (c) EEEM071, University of Surrey

import torch


def init_lr_scheduler(
    optimizer,
    lr_scheduler="multi_step",  # learning rate scheduler
    stepsize=[20, 40],  # step size to decay learning rate
    gamma=0.1,  # learning rate decay
    warmup_epochs=0,  # linear warmup epochs (used by multi_step_warmup / cosine)
    warmup_start_factor=1e-3,  # initial LR multiplier at epoch 0
    max_epoch=None,  # total training epochs (required for cosine T_max)
    cosine_eta_min=0.0,  # final LR floor for cosine annealing
):
    if lr_scheduler == "single_step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize[0], gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step_warmup":
        if warmup_epochs <= 0:
            raise ValueError(
                "multi_step_warmup requires warmup_epochs > 0; "
                "use multi_step if no warmup is wanted."
            )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        decay = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_epochs],
        )

    elif lr_scheduler == "cosine":
        # CLIP-SENet recipe (arXiv 2502.16815 sec IV.B.1, Algorithm 1 line 13).
        # Optional linear warmup followed by cosine annealing to eta_min over the
        # remaining epochs.
        if max_epoch is None or max_epoch <= 0:
            raise ValueError("cosine requires max_epoch > 0 to set T_max.")
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(max_epoch - warmup_epochs, 1),
                eta_min=cosine_eta_min,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epoch, eta_min=cosine_eta_min,
        )

    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")
