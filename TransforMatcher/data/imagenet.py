"""SC-ImageNet semantic correspondence dataset."""
import json
import os

from PIL import Image
import torch

from .dataset import CorrespondenceDataset


class ImageNetDataset(CorrespondenceDataset):
    def __init__(self, benchmark, datapath, thres, split, augment, imside=240):
        super(ImageNetDataset, self).__init__(benchmark, datapath, thres, split, augment, imside)

        with open(self.spt_path) as stream:
            self.train_data = [line.strip() for line in stream if line.strip()]
        self.cls = sorted(name for name in os.listdir(self.img_path)
                          if os.path.isdir(os.path.join(self.img_path, name)))
        class_ids = {name: idx for idx, name in enumerate(self.cls)}
        self.src_bbox, self.trg_bbox = [], []
        self.vpvar, self.scvar, self.trncn, self.occln = [], [], [], []

        for name in self.train_data:
            path = os.path.join(self.ann_path, name + '.json')
            if not os.path.isfile(path):
                path = os.path.join(self.ann_path, name.replace(':', '_') + '.json')
            with open(path) as stream:
                annotation = json.load(stream)

            self.src_imnames.append(annotation['src_imname'])
            self.trg_imnames.append(annotation['trg_imname'])
            self.cls_ids.append(class_ids[annotation['category']])
            self.src_kps.append(torch.tensor(annotation['src_kps']).t().float())
            self.trg_kps.append(torch.tensor(annotation['trg_kps']).t().float())
            self.src_bbox.append(torch.tensor(annotation['src_bndbox']).float())
            self.trg_bbox.append(torch.tensor(annotation['trg_bndbox']).float())
            self.vpvar.append(torch.tensor(annotation['viewpoint_variation']))
            self.scvar.append(torch.tensor(annotation['scale_variation']))
            self.trncn.append(torch.tensor(annotation['truncation']))
            self.occln.append(torch.tensor(annotation['occlusion']))

    def __getitem__(self, idx):
        sample = super(ImageNetDataset, self).__getitem__(idx)
        sample['pckthres'] = self.get_pckthres(sample, sample['trg_imsize'])
        sample['vpvar'] = self.vpvar[idx]
        sample['scvar'] = self.scvar[idx]
        sample['trncn'] = self.trncn[idx]
        sample['occln'] = self.occln[idx]
        sample['class_id'] = self.cls_ids[idx]
        return sample

    def get_image(self, imnames, idx):
        path = os.path.join(self.img_path, self.cls[self.cls_ids[idx]], imnames[idx])
        return Image.open(path).convert('RGB')
