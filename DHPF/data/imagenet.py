r"""SC-ImageNet dataset.

Extract the SC-ImageNet annotations and split lists into SC-IMAGENET.
Download the matching ImageNet images separately and organize the data as follows:

    <datapath>/SC-IMAGENET/
        Layout/large/trn.txt
        Layout/large/val.txt
        PairAnnotation/trn/<pair_id>.json
        PairAnnotation/val/<pair_id>.json
        JPEGImages/<category>/<category>_<image_id>.JPEG

The directory name SC-IMAGENET is case-sensitive on Linux.
For example, JPEGImages/n02138441/n02138441_2292.JPEG is an image path.
Annotation filenames match the split entries directly, as in original DHPF.
"""
import json
import glob
import os

from PIL import Image
import torch

from .dataset import CorrespondenceDataset


class ImageNetDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split):
        r"""SC-ImageNet dataset constructor"""
        super(ImageNetDataset, self).__init__(benchmark, datapath, thres, device, split)

        self.train_data = open(self.spt_path).read().split('\n')
        self.train_data = self.train_data[:len(self.train_data) - 1]
        self.src_imnames = list(map(lambda x: x.split('-')[2].split(':')[1] + '_' + x.split('-')[1] + '.JPEG', self.train_data))
        self.trg_imnames = list(map(lambda x: x.split('-')[2].split(':')[1] + '_' + x.split('-')[2].split(':')[0] + '.JPEG', self.train_data))
        self.cls = os.listdir(self.img_path)
        self.cls.sort()

        anntn_files = []
        for data_name in self.train_data:
            anntn_files.append(glob.glob('%s/%s.json' % (self.ann_path, data_name))[0])
        anntn_files = list(map(lambda x: json.load(open(x)), anntn_files))
        self.src_kps = list(map(lambda x: torch.tensor(x['src_kps']).t().float(), anntn_files))
        self.trg_kps = list(map(lambda x: torch.tensor(x['trg_kps']).t().float(), anntn_files))
        self.src_bbox = list(map(lambda x: torch.tensor(x['src_bndbox']).float(), anntn_files))
        self.trg_bbox = list(map(lambda x: torch.tensor(x['trg_bndbox']).float(), anntn_files))
        self.cls_ids = list(map(lambda x: self.cls.index(x['category']), anntn_files))

        self.vpvar = list(map(lambda x: torch.tensor(x['viewpoint_variation']), anntn_files))
        self.scvar = list(map(lambda x: torch.tensor(x['scale_variation']), anntn_files))
        self.trncn = list(map(lambda x: torch.tensor(x['truncation']), anntn_files))
        self.occln = list(map(lambda x: torch.tensor(x['occlusion']), anntn_files))

    def __getitem__(self, idx):
        r"""Constructs and return a batch for SC-ImageNet dataset"""
        batch = super(ImageNetDataset, self).__getitem__(idx)

        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize'])
        batch['pckthres'] = self.get_pckthres(batch,  batch['trg_imsize'])

        batch['src_kpidx'] = self.match_idx(batch['src_kps'], batch['n_pts'])
        batch['trg_kpidx'] = self.match_idx(batch['trg_kps'], batch['n_pts'])

        batch['vpvar'] = self.vpvar[idx]
        batch['scvar'] = self.scvar[idx]
        batch['trncn'] = self.trncn[idx]
        batch['occln'] = self.occln[idx]

        return batch

    def get_image(self, img_names, idx):
        r"""Returns image tensor"""
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], img_names[idx])

        return Image.open(path).convert('RGB')

    def get_bbox(self, bbox_list, idx, imsize):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        bbox[0::2] *= (self.imside / imsize[0])
        bbox[1::2] *= (self.imside / imsize[1])
        return bbox.to(self.device)
