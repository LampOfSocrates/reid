# Copyright (c) EEEM071, University of Surrey

import datetime
import csv
import json
import os
import os.path as osp
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from src import models
from src.data_manager import ImageDataManager
from src.eval_metrics import evaluate
from src.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from src.lr_schedulers import init_lr_scheduler
from src.optimizers import init_optimizer
from src.utils.avgmeter import AverageMeter
from src.utils.generaltools import set_random_seed
from src.utils.iotools import check_isfile
from src.utils.loggers import Logger, RankLogger
from src.utils.torchtools import (
    count_num_param,
    accuracy,
    load_pretrained_weights,
    save_checkpoint,
    resume_from_checkpoint,
)
from src.utils.visualtools import visualize_ranked_results
import uuid

# global variables
parser = argument_parser()
args = parser.parse_args()
_REQUESTED_SAVE_DIR = args.save_dir
_RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
args.save_dir = f"{args.save_dir}_{_RUN_TIMESTAMP}"
_COMPARISON_EPOCHS = []
_COMPARISON_EVALS = []


def _write_comparison_outputs(total_elapsed_seconds):
    comparison_json = osp.join(args.save_dir, "comparison_summary.json")
    comparison_csv = osp.join(args.save_dir, "comparison_summary.csv")

    avg_epoch_time = (
        sum(item["epoch_time_seconds"] for item in _COMPARISON_EPOCHS) / len(_COMPARISON_EPOCHS)
        if _COMPARISON_EPOCHS
        else 0.0
    )

    best_rank1 = None
    best_map = None
    final_eval = None
    if _COMPARISON_EVALS:
        final_eval = _COMPARISON_EVALS[-1]
        best_rank1 = max(item["rank1"] for item in _COMPARISON_EVALS)
        best_map = max(item["mAP"] for item in _COMPARISON_EVALS)

    summary = {
        "requested_save_dir": _REQUESTED_SAVE_DIR,
        "run_save_dir": args.save_dir,
        "timestamp": _RUN_TIMESTAMP,
        "arch": args.arch,
        "source_names": list(args.source_names),
        "target_names": list(args.target_names),
        "root": args.root,
        "height": args.height,
        "width": args.width,
        "optim": args.optim,
        "lr": args.lr,
        "max_epoch": args.max_epoch,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "train_sampler": args.train_sampler,
        "data_fraction": getattr(args, "data_fraction", 1.0),
        "total_elapsed_seconds": total_elapsed_seconds,
        "avg_epoch_time_seconds": avg_epoch_time,
        "best_rank1": best_rank1,
        "best_mAP": best_map,
        "final_eval": final_eval,
        "epoch_summaries": _COMPARISON_EPOCHS,
        "eval_history": _COMPARISON_EVALS,
    }

    with open(comparison_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(comparison_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "arch",
                "source_names",
                "target_names",
                "requested_save_dir",
                "run_save_dir",
                "lr",
                "max_epoch",
                "train_batch_size",
                "test_batch_size",
                "train_sampler",
                "data_fraction",
                "total_elapsed_seconds",
                "avg_epoch_time_seconds",
                "best_rank1",
                "best_mAP",
                "final_rank1",
                "final_mAP",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": _RUN_TIMESTAMP,
                "arch": args.arch,
                "source_names": ",".join(args.source_names),
                "target_names": ",".join(args.target_names),
                "requested_save_dir": _REQUESTED_SAVE_DIR,
                "run_save_dir": args.save_dir,
                "lr": args.lr,
                "max_epoch": args.max_epoch,
                "train_batch_size": args.train_batch_size,
                "test_batch_size": args.test_batch_size,
                "train_sampler": args.train_sampler,
                "data_fraction": getattr(args, "data_fraction", 1.0),
                "total_elapsed_seconds": total_elapsed_seconds,
                "avg_epoch_time_seconds": avg_epoch_time,
                "best_rank1": best_rank1,
                "best_mAP": best_map,
                "final_rank1": final_eval["rank1"] if final_eval else None,
                "final_mAP": final_eval["mAP"] if final_eval else None,
            }
        )

    print(f'Comparison summary saved to "{comparison_json}"')
    print(f'Comparison summary saved to "{comparison_csv}"')


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = "log_test.txt" if args.evaluate else "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========")
    student_id = os.environ.get('STUDENT_ID', '<your id>')
    student_name = os.environ.get('STUDENT_NAME', '<your name>')
    print("Student ID:{}".format(student_id))
    print("Student name:{}".format(student_name))
    print("UUID:{}".format(uuid.uuid4()))
    print("Experiment time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print("==========")
    print(f"==========\nArgs:{args}\n==========")
    print(f"Requested save dir:{_REQUESTED_SAVE_DIR}")
    print(f"Run save dir:{args.save_dir}")

    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print(f"Initializing model: {args.arch}")
    model = models.init_model(
        name=args.arch,
        num_classes=dm.num_train_pids,
        loss={"xent", "htri"},
        pretrained=not args.no_pretrained,
        use_gpu=use_gpu,
    )
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion_xent = CrossEntropyLoss(
        num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth
    )
    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print(f"Evaluating {name} ...")
            queryloader = testloader_dict[name]["query"]
            galleryloader = testloader_dict[name]["gallery"]
            distmat = test(
                model, queryloader, galleryloader, use_gpu, return_distmat=True
            )

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat,
                    dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, "ranked_results", name),
                    topk=20,
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print("=> Start training")
    """
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    """
    for epoch in range(args.start_epoch, args.max_epoch):
        epoch_stats = train(
            epoch,
            model,
            criterion_xent,
            criterion_htri,
            optimizer,
            trainloader,
            use_gpu,
        )
        _COMPARISON_EPOCHS.append(epoch_stats)
        print(
            f"Epoch {epoch + 1}/{args.max_epoch} finished in "
            f"{epoch_stats['epoch_time_seconds']:.2f}s"
        )

        scheduler.step()

        if (
            (epoch + 1) > args.start_eval
            and args.eval_freq > 0
            and (epoch + 1) % args.eval_freq == 0
            or (epoch + 1) == args.max_epoch
        ):
            print("=> Test")

            for name in args.target_names:
                print(f"Evaluating {name} ...")
                queryloader = testloader_dict[name]["query"]
                galleryloader = testloader_dict[name]["gallery"]
                eval_metrics = test(
                    model,
                    queryloader,
                    galleryloader,
                    use_gpu,
                    return_metrics=True,
                )
                ranklogger.write(name, epoch + 1, eval_metrics["rank1"])
                _COMPARISON_EVALS.append(
                    {
                        "epoch": epoch + 1,
                        "dataset": name,
                        **eval_metrics,
                    }
                )

            if args.save_checkpoint:
                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "rank1": eval_metrics["rank1"],
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "optimizer": optimizer.state_dict(),
                    },
                    args.save_dir,
                )

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Elapsed {elapsed}")
    _write_comparison_outputs(time.time() - time_start)
    ranklogger.show_summary()


def train(
    epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu
):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    for p in model.parameters():
        p.requires_grad = True  # open all layers

    epoch_start = time.time()
    end = time.time()
    progress = tqdm(
        enumerate(trainloader),
        total=len(trainloader),
        desc=f"Epoch {epoch + 1}/{args.max_epoch}",
        file=sys.stdout,
        leave=True,
    )
    for batch_idx, (imgs, pids, _, _) in progress:
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs, features = model(imgs)
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(trainloader):
            progress.set_postfix(
                time=f"{batch_time.avg:.3f}s",
                data=f"{data_time.avg:.4f}s",
                xent=f"{xent_losses.avg:.4f}",
                htri=f"{htri_losses.avg:.4f}",
                acc=f"{accs.avg:.2f}",
            )

        end = time.time()

    return {
        "epoch": epoch + 1,
        "epoch_time_seconds": time.time() - epoch_start,
        "avg_batch_time_seconds": batch_time.avg,
        "avg_data_time_seconds": data_time.avg,
        "avg_xent": xent_losses.avg,
        "avg_htri": htri_losses.avg,
        "avg_acc": accs.avg,
    }


def test(
    model,
    queryloader,
    galleryloader,
    use_gpu,
    ranks=[1, 5, 10, 20],
    return_distmat=False,
    return_metrics=False,
):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print(
            "Extracted features for query set, obtained {}-by-{} matrix".format(
                qf.size(0), qf.size(1)
            )
        )

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print(
            "Extracted features for gallery set, obtained {}-by-{} matrix".format(
                gf.size(0), gf.size(1)
            )
        )

    print(
        f"=> BatchTime(s)/BatchSize(img): {batch_time.avg:.3f}/{args.test_batch_size}"
    )

    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print(f"mAP: {mAP:.1%}")
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    if return_metrics:
        return {
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]) if len(cmc) >= 5 else None,
            "rank10": float(cmc[9]) if len(cmc) >= 10 else None,
            "rank20": float(cmc[19]) if len(cmc) >= 20 else None,
            "mAP": float(mAP),
        }
    return cmc[0]


if __name__ == "__main__":
    main()
