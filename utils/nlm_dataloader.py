# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from pymongo import MongoClient
from functools import lru_cache


import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from nlm_utils.storage import file_storage

from utils.general import (
    check_requirements,
    xyxy2xywh,
    xywh2xyxy,
    xywhn2xyxy,
    xyn2xy,
    segment2box,
    segments2boxes,
    resample_segments,
    clean_str,
)
from utils.torch_utils import torch_distributed_zero_first


logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    opt,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    world_size=1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=opt.single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
        )

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers]
    )  # number of workers
    nw = 1
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    )
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn4
        if quad
        else LoadImagesAndLabels.collate_fn,
    )
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def xyxy_to_training(xyxy, dim):
    x1, y1, x2, y2 = xyxy

    width = (x2 - x1) / dim[1]
    height = (y2 - y1) / dim[0]
    midX = (x1 + x2) / 2 / dim[1]
    midY = (y1 + y2) / 2 / dim[0]

    return [midX, midY, width, height]


@lru_cache(maxsize=1024)
def load_json(file_idx, cache_folder="./nlm_features/"):
    filename = f"{cache_folder}/{file_idx}.json"
    if not os.path.exists(filename):
        file_storage.download_document(
            f"bbox/features/{file_idx}.json",
            dest_file_location=f"{cache_folder}/{file_idx}.json",
        )

    with open(str(filename)) as f:
        data = json.load(f)
    return data


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(
        self,
        path,
        img_size=1344,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights

        self.stride = stride
        self.path = path
        self.imgs = []
        self.img_files = []
        self.labels = []
        self.dimensions = 0

        self.labels_map = {"table": 0}

        db_client = MongoClient(os.getenv("MONGO_HOST", "localhost"))
        db = db_client[os.getenv("MONGO_DATABASE", "doc-store-dev")]

        file_idxs = db["bboxes"].distinct(
            "file_idx", {"audited": True, "block_type": "table"}
        )

        for file_idx in file_idxs:
            data = load_json(file_idx)

            # self.img_names.append(str(filename.name).split(".json")[0])
            if not self.dimensions:
                self.dimensions = len(data["metadata"]["features"]) + 1
            else:
                assert self.dimensions == len(data["metadata"]["features"]) + 1

            pages = db["bboxes"].distinct(
                "page_idx",
                {
                    "file_idx": file_idx,
                    "audited": True,
                    "block_type": "table",
                },
            )
            for page_idx in pages:
                tokens = data["data"][page_idx]

                # HWC.
                # We hard code the padding to 0, thus image must in square (H:1344,W:1344 by default)
                features = np.zeros(
                    (self.img_size, self.img_size, self.dimensions),
                    dtype=np.float32,
                )

                # make features, HWF
                for token in tokens:
                    # x1, y1, x2, y2 = xyxy_to_training(token["position"]["xyxy"])
                    x1, y1, x2, y2 = token["position"]["xyxy"]

                    # token position mask
                    features[int(y1) : int(y2), int(x1) : int(x2), 0] = 1

                    for i, feature in enumerate(token["features"]):
                        features[int(y1) : int(y2), int(x1) : int(x2), i + 1] = feature

                self.imgs.append(features)

                self.img_files.append(f"file:{file_idx} page:{page_idx+1}")

                # make labels
                page_labels = []
                for label in db["bboxes"].find(
                    {
                        "file_idx": file_idx,
                        "page_idx": page_idx,
                        "audited": True,
                        "block_type": "table",
                    }
                ):
                    label_type = self.labels_map[label["block_type"]]
                    label_coords = xyxy_to_training(
                        label["bbox"], dim=(self.img_size, self.img_size)
                    )

                    labeled = [label_type] + label_coords

                    page_labels.append(np.array(labeled))

                print(
                    f"{len(page_labels)} labels for document {file_idx}, page {page_idx+1}"
                )
                self.labels.append(page_labels)

        # label found, label missing, label empty, label corrupted, total label.
        self.shapes = np.array(
            [[x.shape[0], x.shape[1]] for x in self.imgs], dtype=np.float64
        )
        # convert to np array
        self.labels = [np.array(x) for x in self.labels]

        n = len(self.imgs)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        img = self.imgs[index]

        # create shapes for plotting
        h0, w0 = h, w = self.img_size, self.img_size

        # orignal (h,w), (scale(h,w), padding(h,w)).
        # NOTE: padding is hard coded to 0, thus we should have
        # shape = (1344, 1344), ((1, 1), (0, 0))
        shapes = (h0, w0), ((h / h0, w / w0), (0, 0))  # for COCO mAP rescaling

        labels = self.labels[index]
        nL = len(labels)  # number of labels
        labels_out = torch.zeros((nL, 6))
        labels_out[:, 1:] = torch.from_numpy(labels)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(
                    img[i].unsqueeze(0).float(),
                    scale_factor=2.0,
                    mode="bilinear",
                    align_corners=False,
                )[0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat(
                    (
                        torch.cat((img[i], img[i + 1]), 1),
                        torch.cat((img[i + 2], img[i + 3]), 1),
                    ),
                    2,
                )
                l = (
                    torch.cat(
                        (
                            label[i],
                            label[i + 1] + ho,
                            label[i + 2] + wo,
                            label[i + 3] + ho + wo,
                        ),
                        0,
                    )
                    * s
                )
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


class LoadNLMFeatures:  # for inference
    def __init__(self, img_size=1344, stride=32):
        self.img_size = img_size

        self.mode = "image"
        self.cap = None
        self.dimensions = 0
        self.imgs = []
        self.img_files = []

        db_client = MongoClient(os.getenv("MONGO_HOST", "localhost"))
        db = db_client[os.getenv("MONGO_DATABASE", "doc-store-dev")]

        file_idxs = db["bboxes"].distinct("file_idx", {})

        for file_idx in file_idxs:
            data = load_json(file_idx)

            # self.img_names.append(str(filename.name).split(".json")[0])
            if not self.dimensions:
                self.dimensions = len(data["metadata"]["features"]) + 1
            else:
                assert self.dimensions == len(data["metadata"]["features"]) + 1

            pages = db["bboxes"].distinct(
                "page_idx",
                {
                    "file_idx": file_idx,
                },
            )
            
            for page_idx in pages:
                tokens = data["data"][page_idx]

                # HWC.
                # We hard code the padding to 0, thus image must in square (H:1344,W:1344 by default)
                features = np.zeros(
                    (self.img_size, self.img_size, self.dimensions),
                    dtype=np.float32,
                )

                # make features, HWF
                for token in tokens:
                    # x1, y1, x2, y2 = xyxy_to_training(token["position"]["xyxy"])
                    x1, y1, x2, y2 = token["position"]["xyxy"]

                    # token position mask
                    features[int(y1) : int(y2), int(x1) : int(x2), 0] = 1

                    for i, feature in enumerate(token["features"]):
                        features[int(y1) : int(y2), int(x1) : int(x2), i + 1] = feature

                self.imgs.append(features)

                self.img_files.append(f"{file_idx}_{page_idx+1}.jpg")

                print(f"features loaded for document {file_idx}, page {page_idx+1}")


    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.imgs):
            raise StopIteration

        # Read image
        self.count += 1
        img = self.imgs[self.count - 1]

        # NLM features in BHWC, yolo image in BCHW
        # # HWC => CWH
        # img0 = img.transpose(2, 0, 1)

        # use top-3 channel as image
        img0 = img[:, :, :3] * 200

        # scale to 0-255
        img0[img0 < 0] = 0
        img0[img0 > 255] = 255

        img0 = img0.astype(np.uint8)

        path = self.img_files[self.count - 1]
        print(f"image {self.count}/{len(self.imgs)} {path}: ", end="")

        return path, img, img0, self.cap