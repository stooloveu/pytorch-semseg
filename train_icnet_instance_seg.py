import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from torch.utils import data
from tqdm import tqdm


from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loss.loss import multi_scale_discrimitive_loss_cs
from ptsemseg.loss.loss import multi_scale_cross_entropy2d_inst
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict

from tensorboardX import SummaryWriter

class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = multi_scale_cross_entropy2d_inst
        self.loss_d = multi_scale_discrimitive_loss_cs

    def forward(self, targets, targets_inst, *inputs, return_aux_info = False):
        # print(inputs.get_device())
        # print(targets.get_device())
        # if self.eval:
        #     import ipdb
        #     ipdb.set_trace() 
        outputs = self.model(*inputs)
        # if not no_ref:
        if self.training:
            loss = self.loss(outputs[0:3], targets)
            loss_d = self.loss_d(outputs[3:6], targets_inst)
        else:
            loss = self.loss(outputs[0], targets)
            loss_d = self.loss_d(outputs[1], targets_inst)

        if return_aux_info:
            aux_info = (torch.unsqueeze(loss, 0), torch.unsqueeze(loss_d, 0))
            return torch.unsqueeze(0 * loss + loss_d, 0), outputs, aux_info
            # return torch.unsqueeze(loss_d, 0), outputs, aux_info
        else:
            return torch.unsqueeze(0 * loss + loss_d, 0), outputs
            # return torch.unsqueeze(loss_d, 0), outputs
        # else:
        #     return outputs
        

def DataParallel_withLoss(model,loss,**kwargs):
    model=FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None
    if 'output_device' in kwargs.keys():
        output_device=kwargs['output_device']
    else:
        output_device=None
    if 'cuda' in kwargs.keys():
        cudaID=kwargs['cuda'] 
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
    return model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
            return param_group['lr']

def train(cfg, writer, logger):

    # Setup seeds
    # torch.manual_seed(cfg.get("seed", 1337))
    # torch.cuda.manual_seed(cfg.get("seed", 1337))
    # np.random.seed(cfg.get("seed", 1337))
    # random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg["model"], n_classes).to(device)

    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))


   
    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            
            if not args.load_weight_only:
                model = DataParallel_withLoss(model,loss_fn)
                model.load_state_dict(checkpoint["model_state"])
                if not args.not_load_optimizer:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])

                # !!!
                # checkpoint["scheduler_state"]['last_epoch'] = -1
                # scheduler.load_state_dict(checkpoint["scheduler_state"])
                # start_iter = checkpoint["epoch"]
                start_iter = 0
                # import ipdb
                # ipdb.set_trace()
                logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
            else:
                pretrained_dict = convert_state_dict(checkpoint["model_state"])
                model_dict = model.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 
                # 3. load the new state dict
                model.load_state_dict(model_dict)
                model = DataParallel_withLoss(model,loss_fn)
                # import ipdb
                # ipdb.set_trace()
                # start_iter = -1
                logger.info(
                "Loaded checkpoint '{}' (iter unknown, from pretrained icnet model)".format(
                    cfg["training"]["resume"]
                )
                )

        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True



    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels, inst_labels) in trainloader:

            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            inst_labels = inst_labels.to(device)
            optimizer.zero_grad()

            loss, _, aux_info = model(labels, inst_labels, images, return_aux_info = True)
            loss = loss.sum()
            loss_sem = aux_info[0].sum()
            loss_inst = aux_info[1].sum()
            
            # loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f} (Sem:{:.4f}/Inst:{:.4f})  LR:{:.5f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    loss_sem.item(),
                    loss_inst.item(),
                    scheduler.get_lr()[0],
                    time_meter.avg / cfg["training"]["batch_size"],
                )
                
                # print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:

                model.eval()

                with torch.no_grad():
                    for i_val, (images_val, labels_val, inst_labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        inst_labels_val = inst_labels_val.to(device)
                        # outputs = model(images_val)
                        # val_loss = loss_fn(input=outputs, target=labels_val)
                        val_loss, (outputs, outputs_inst) = model(labels_val, inst_labels_val, images_val)
                        val_loss = val_loss.sum()

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

            if (i + 1) % cfg["training"]["save_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                state = {
                    "epoch": i + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_iou": best_iou,
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_{}_{:05d}_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"], i+1),
                )
                torch.save(state, save_path)

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--load_weight_only",
        dest="load_weight_only",
        action="store_true",
        help="Only load weight from saved models |\
                              False by default",
    )
    parser.set_defaults(load_weight_only=False)
    parser.add_argument(
        "--not_load_optimizer",
        dest="not_load_optimizer",
        action="store_true",
        help="Do not load optimizer |\
                              False by default",
    )
    parser.set_defaults(load_weight_only=False)
    parser.set_defaults(not_load_optimizer=False)
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)
