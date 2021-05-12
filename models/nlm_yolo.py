# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path


sys.path.append(
    Path(__file__).parent.parent.absolute().__str__()
)  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from models.yolo import Model as YoloModel
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import (
    time_synchronized,
    fuse_conv_and_bn,
    model_info,
    scale_img,
    initialize_weights,
    select_device,
    copy_attr,
)

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (
                        y[..., 0:2] * 2.0 - 0.5 + self.grid[i]
                    ) * self.stride[
                        i
                    ]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                        1, self.na, 1, 1, 2
                    )  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg, num_features=11, channels=3, num_classes=1, **kwargs):
        super().__init__()

        self.num_features = num_features
        self.channels = channels
        self.num_classes = num_classes

        kwargs.pop("ch")
        kwargs.pop("nc")

        self.yolo = YoloModel(cfg, ch=channels, nc=num_classes, **kwargs)
        # self.yolo = self.yolo.autoshape()
        
        self.stride =self.yolo.stride
        self.model = self.yolo.model

        # BHWC
        self.feature_w = nn.Linear(num_features, channels)

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs, **kwargs):
        for img in imgs:
            assert img.shape[0] % 32 == img.shape[1] % 32 == 0

        assert imgs.shape[-1] == self.num_features

        x = imgs

        # linear layer
        x = self.feature_w(x)

        # sigmoid
        x = self.sigmoid(x)

        # BHWC to BCHW
        x = x.transpose(1, 3)

        # run yolo
        return self.yolo(x)

    def parse_yolo_output(self, logits, confident_threshold=0.5, iou_threshold=0.5):
        # Post-process
        preds = non_max_suppression(
            logits,
            conf_thres=confident_threshold,
            iou_thres=iou_threshold,
            classes=range(self.num_classes),
        )

        return preds


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info(
        "\n%3s%18s%3s%10s  %-40s%-30s"
        % ("", "from", "n", "params", "module", "arguments")
    )
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
        ]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        logger.info("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
