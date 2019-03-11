import os
import torch
import numpy as np
import scipy.misc as m
import imageio

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class cityscapesInstWpLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
            self,
            root,
            split="train",
            is_transform=False,
            img_size=(512, 1024),
            augmentations=None,
            img_norm=False,
            version="pascal",
            test_mode=False,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size,
                                                                      img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        if not test_mode:
            self.images_base = os.path.join(self.root, "leftImg8bit",
                                            self.split)
            self.annotations_base = os.path.join(self.root, "gtFine",
                                                 self.split)

            self.files[split] = recursive_glob(
                rootdir=self.images_base, suffix=".png")

        self.void_classes = [
            0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1
        ]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.has_instance_classes = [
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not test_mode:
            if not self.files[split]:
                raise Exception("No files for split=[%s] found in %s" %
                                (split, self.images_base))

            print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        inst_lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png",
        )

        img = imageio.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = imageio.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        inst_lbl = m.imread(inst_lbl_path)
        inst_lbl = np.array(inst_lbl, dtype=np.int32)
        # inst_lbl = self.encode_inst_segmap(np.array(inst_lbl, dtype=np.uint16))

        if self.augmentations is not None:
            img, lbl, inst_lbl = self.augmentations(img, lbl, inst_lbl)

        if self.is_transform:
            img, lbl, inst_lbl = self.transform(img, lbl, inst_lbl)

        # return img, lbl, self.get_tot_inst_num(inst_lbl)
        return img, lbl, inst_lbl

    def transform(self, img, lbl, inst_lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        H = img.shape[1]
        W = img.shape[2]

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(
            lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        inst_lbl = inst_lbl.astype(float)
        inst_lbl = m.imresize(
            inst_lbl, (self.img_size[0], self.img_size[1]),
            "nearest",
            mode="F")
        inst_lbl = inst_lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(
                np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        # position_y = np.transpose(np.tile(np.linspace(-1.0, 1.0, H),
        #                                   (W, 1))).reshape(1, H, W)
        # position_x = np.tile(np.linspace(-1.0, 1.0, W), (H, 1)).reshape(
        #     1, H, W)
        # img_coord = np.concatenate((img, position_y, position_x), axis=0)

        # img = torch.from_numpy(img_coord).float()
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        inst_lbl = torch.from_numpy(inst_lbl).long()
        return img, lbl, inst_lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def encode_inst_segmap(self, mask):
        # inst_num = self.get_tot_inst_num(mask)
        inst_mask = []
        # inst_layer_id = 0
        for _inst_sem_id in self.has_instance_classes:
            if not mask[mask // 1000 == _inst_sem_id].size == 0:
                for i in range(
                        np.max(mask[mask // 1000 == _inst_sem_id] % 1000) + 1):
                    inst_mask_single = np.zeros(mask.shape, dtype=np.uint8)
                    inst_mask_single[mask == _inst_sem_id * 1000 + i] = True
                    inst_mask.append(inst_mask_single)
        inst_mask = np.stack(inst_mask, axis=0)
        return inst_mask

    def get_tot_inst_num(self, mask):
        inst_num = 0
        for _inst_sem_id in self.has_instance_classes:
            if not mask[mask // 1000 == _inst_sem_id].size == 0:
                inst_num += np.max(
                    mask[mask // 1000 == _inst_sem_id] % 1000) + 1
                ipdb.set_trace()
        return inst_num

    def get_instance_pos(self, inst_lbl):
        '''
        For each pixel, get its corresponding object's center/topleft corner/bottomright corner coordinates
        Stuff classes are set to 0.
        input: inst_lbl [H,W] with 23001 like
        return: tensor [H,W,6(c_y,c_x,tl_y,tl_x,br_y,br_x)]
        '''

        H = inst_lbl.shape[0]
        W = inst_lbl.shape[1]
        instance_pos = np.zeros((H, W, 6))
        instances = np.isin(inst_lbl // 1000,
                            self.has_instance_classes).nonzero()
        for i in np.unique(instances):
            instance = (inst_lbl == i).nonzero()
            instance_pos[instance[0], instance[1], 0] = np.mean(
                instance[0]) / H * 2 - 1  # center_y
            instance_pos[instance[0], instance[1], 1] = np.mean(
                instance[1]) / W * 2 - 1  # center_x
            instance_pos[instance[0], instance[1], 2] = np.amin(
                instance[0]) / H * 2 - 1  # tl_y
            instance_pos[instance[0], instance[1], 3] = np.amin(
                instance[1]) / W * 2 - 1 / W * 2 - 1  # tl_x
            instance_pos[instance[0], instance[1], 4] = np.amax(
                instance[0]) / H * 2 - 1  # br_y
            instance_pos[instance[0], instance[1], 5] = np.amax(
                instance[1]) / W * 2 - 1  # br_x

        return torch.from_numpy(instance_pos).float()


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    # local_path = "/datasets01/cityscapes/112817/"
    # dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    # bs = 4
    # trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    # for i, data_samples in enumerate(trainloader):
    #     imgs, labels = data_samples
    #     import pdb

    #     pdb.set_trace()
    #     imgs = imgs.numpy()[:, ::-1, :, :]
    #     imgs = np.transpose(imgs, [0, 2, 3, 1])
    #     f, axarr = plt.subplots(bs, 2)
    #     for j in range(bs):
    #         axarr[j][0].imshow(imgs[j])
    #         axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
    #     plt.show()
    #     a = input()
    #     if a == "ex":
    #         break
    #     else:
    #         plt.close()
    import ipdb
    local_path = "/scratch/xiac/pytorch-semseg/datasets/cityscapes"
    dst = cityscapesLoader(local_path)
    bs = 16
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=16)
    max_inst_num = 0
    for i, data_samples in enumerate(trainloader):
        imgs, labels, inst_labels = data_samples
        # max_inst_num = max(torch.max(inst_num), max_inst_num)
        print(i)
    #     ipdb.set_trace()
    # print(max_inst_num)
    ipdb.set_trace()