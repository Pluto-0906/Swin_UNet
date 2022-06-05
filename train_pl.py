import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split

from custom_loss.dice_score import dice_loss
from networks.vision_transformer import SwinUnet as ViT_seg
from datasets.dataset_spine import Spine_Dataset
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Spine', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--output_dir', type=str,
                    default='/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/output', help='output dir')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--val_percent', type=float,
                    default=0.1, help='val_set percent')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
                    metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs.",
    default=None,
    nargs='+',
)
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


class YasuoModel(pl.LightningModule):
    def __init__(self, config, args, net, threshold=0.5, **kwargs):
        super().__init__()
        self.config = config
        self.args = args
        self.model = net # TODO
        # import pdb;pdb.set_trace()
        self.loss_fn = dice_loss()
        self.threshold = threshold
        self.base_lr = args.base_lr

    def forward(self, img):
        output = self.model(img)
        output = F.sigmoid(output)
        return output

    def configure_optimizers(self):
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=0.0001)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=250, T_mult=2, eta_min=0, last_epoch=-1
        )
        return [optimizer], [scheduler]

    def shared_step(self, batch, stage):

        img = batch["img"]
        assert img.ndim == 4

        h, w = img.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt = batch["label"]
        assert gt.ndim == 4

        output = self.forward(img)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(output, gt)

        pred_mask = (output > self.threshold).type(torch.uint8)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), gt.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        prec = smp.metrics.precision(
            tp, fp, fn, tn, reduction="micro-imagewise")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, "micro-imagewise")

        metrics = {
            # f"{stage}_per_image_iou": per_image_iou,
            # f"{stage}_dataset_iou": dataset_iou,
            f"{stage}rec": rec,
            f"{stage}prec": prec,
            f"{stage}f1": f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        # return self.shared_epoch_end(outputs, "train")
        pass

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")


args = parser.parse_args()
config = get_config(args)

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    images_dir = args.root_path + 'imgs/'
    labels_dir = args.root_path + 'masks/'

    # 1. create dataset
    dataset = Spine_Dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        augmentation=True,
    )

    # 2. split dataset
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloader
    loader_args = dict(batch_size=args.batch_size,
                       num_workers=16, pin_memory=True)
    # type: ignore os.cpu_count()
    train_dataloader = DataLoader(
        dataset=train_set, shuffle=True, **loader_args)
    val_dataloader = DataLoader(dataset=val_set, shuffle=False, **loader_args)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    net.load_from(config)
    # 4. create a model
    model = YasuoModel(config, args, net)

    # 5. define a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs)

    # 6. train the network
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
