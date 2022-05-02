from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import pandas as pd

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

from utils import xywh_to_xyxy, get_iou, norm_to_reg
from refcoco_utils import REFER


project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent

dataset_dir = workspace_dir.joinpath('datasets/').resolve()
shape_dataset_dir = workspace_dir.joinpath('vl_finetune/').resolve()
coco_dir = dataset_dir.joinpath('COCO')

coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')

refcoco_dir = dataset_dir.joinpath('RefCOCO')
refcocog_feature_dir = refcoco_dir.joinpath('refcocog/features')

# Our dataset location

shape_train_img_GT = shape_dataset_dir.joinpath('train_datainfo.csv')
shape_train_img_features = shape_dataset_dir.joinpath('train_dict.pkl')
shape_train_img_dir = shape_dataset_dir.joinpath('train_finetune_vl/')

shape_val_img_GT = shape_dataset_dir.joinpath('val_datainfo.csv')
shape_val_img_features = shape_dataset_dir.joinpath('val_dict.pkl')
shape_val_img_dir = shape_dataset_dir.joinpath('val_finetune_vl/')

shape_test_img_GT = shape_dataset_dir.joinpath('test_datainfo.csv')
shape_test_img_features = shape_dataset_dir.joinpath('test_dict.pkl')
shape_test_img_dir = shape_dataset_dir.joinpath('test_finetune_vl/')

class RefCOCOFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.split = split
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(args.backbone)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(args.backbone)

        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        # mattnet_maskrcnn_detections_path = refcoco_dir.joinpath(
        #     'detections/refcocog_umd/res101_coco_minus_refer_notime_dets.json')
        # with open(mattnet_maskrcnn_detections_path) as f:
        #     mattnet_maskrcnn_detections = json.load(f)

        data = []
        self.refer = REFER('refcocog', 'umd', img_dir=coco_img_dir, ref_dir=refcoco_dir, verbose=verbose)
        ref_ids = self.refer.getRefIds(split=split)

        for ref_id in ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                data.append(
                    {
                        "caption": caption,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        "ref_id": ref_id,
                    }
                )

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes


        self.source_to_h5 = {
            'train': refcocog_feature_dir.joinpath(f'train_boxes_GT.h5')
        }

        if self.args.RefCOCO_GT:
            self.source_to_h5['val'] = refcocog_feature_dir.joinpath(f'val_boxes_GT.h5')
            self.source_to_h5['test'] = refcocog_feature_dir.joinpath(f'test_boxes_GT.h5')
        else:
            self.source_to_h5['val'] = refcocog_feature_dir.joinpath(f'val_boxes_mattnet.h5')
            self.source_to_h5['test'] = refcocog_feature_dir.joinpath(f'test_boxes_mattnet.h5')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        # uid = datum['uid']
        # out_dict['uid'] = uid

        # test = 'test' in datum['annot_id']
        # out_dict['is_test'] = test

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['image_id']
            out_dict['img_id'] = img_id

            # img_path = coco_img_dir.joinpath(datum['img_fn'])
            # assert img_path.exists()
            # out_dict['img_path'] = img_path

            # source = self.img_ids_to_source[img_id]
            source = self.split

            f = self.source_to_h5[source]

            if isinstance(f, Path):
                f = h5py.File(f, 'r')
                self.source_to_h5[source] = f

            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]

            # pred_boxes = f[f'{img_id}/boxes']

            boxes = f[f'{img_id}/boxes'][:self.args.n_boxes]

            # shuffle box order
            if self.args.shuffle_boxes and self.mode == 'train':
                box_indices = np.arange(len(boxes))
                np.random.shuffle(box_indices)

                boxes = boxes[box_indices]

            n_boxes = len(boxes)

            out_dict['n_boxes'] = n_boxes

            ref_box = datum['refBox']

            ref_box = xywh_to_xyxy(np.array([ref_box], dtype=np.float32))
            
            # print(type(ref_box))
            # print(ref_box.dtype)
            # print("Here is ref_box.shape", ref_box.shape)
            # print("Here is boxes.shape", boxes.shape)
            print(ref_box)
            print(boxes)

            ious = get_iou(
                torch.tensor(boxes, dtype=torch.float),
                torch.tensor(ref_box, dtype=torch.float))
            
            # print("Here is ious", ious)

            threshold = 0.5
            scores = ious.detach().numpy().flatten()
            scores[scores < threshold] = 0
            scores = scores.astype(np.float64)
            # print(scores)

            exists_target = scores.sum() > 0

            if exists_target:
                correct_indices = np.nonzero(scores)[0].tolist()
                prob = scores / scores.sum()
                # print("Here is prob", prob)
                # if np.count_nonzero(prob) > 1:
                #     print('noooooo')

                choice = np.random.multinomial(1, prob)
                # print("Here is np.random.multinomial(1, prob)", choice)
                choice = choice.argmax()
                # print("Here is choice", choice)
            else:
                correct_indices = []
                choice = -100

            # Normalize the boxes (to 0 ~ 1)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h

            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            # assert boxes.size() == (36, 4), (boxes.size(),
            #                                  datum['img_id'], gt_boxes.shape, pred_boxes.shape)

            boxes.clamp_(min=0.0, max=1.0)

            out_dict['boxes'] = boxes

            feats = f[f'{img_id}/features'][:self.args.n_boxes]

            if self.args.shuffle_boxes and self.mode == 'train':
                feats = feats[box_indices]

            feats = torch.from_numpy(feats)
            # print(feats.shape)

            out_dict['vis_feats'] = feats
            out_dict['boxes'] = boxes

        ###### Text #####x

        sent = datum['caption']
        # print(type(sent))

        # prefix = "refer expressions:"
        prefix = "visual grounding:"
        # prefix = "grounding:"
        input_text = f'{prefix} {sent}'
        # print(type(input_text))

        if exists_target:
            if self.args.vis_pointer:
                all_target_ids = correct_indices
                target_text = ''
            else:
                target_text = f'<vis_extra_id_{choice}>'
                all_target_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{idx}>' for idx in correct_indices])

        else:
            if self.args.vis_pointer:
                all_target_ids = []
                target_text = ''
            else:
                target_text = ''
                all_target_ids = []

        out_dict['exists_target'] = exists_target
        out_dict['iou'] = ious
        out_dict['target'] = choice
        out_dict['all_targets'] = correct_indices
        out_dict['all_target_ids'] = all_target_ids

        input_ids = self.tokenizer.encode(input_text, max_length=self.args.max_text_length, truncation=True)
        target_ids = self.tokenizer.encode(target_text, max_length=self.args.max_text_length, truncation=True)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['input_text'] = input_text
        out_dict['target_text'] = target_text


        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = self.args

        B = len(batch)

        if args.use_vision:
            V_L = max([b['n_boxes'] for b in batch])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        exists_target = torch.zeros(B, dtype=torch.float)

        input_texts = []
        target_texts = []
        targets = []
        all_targets = []
        all_target_ids = []
        ious = []
        img_ids = []

        for i, entry in enumerate(batch):

            if args.use_vision:
                boxes[i, :entry['n_boxes']] += entry['boxes']
                vis_feats[i, :entry['n_boxes']] += entry['vis_feats']

                vis_attention_mask[i, :entry['n_boxes']] = 1

            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            input_texts.append(entry['input_text'])
            target_texts.append(entry['target_text'])
            targets.append(entry['target'])
            all_targets.append(entry['all_targets'])
            all_target_ids.append(entry['all_target_ids'])
            # ious.append(entry['iou'])
            img_ids.append(entry['img_id'])


            exists_target[i] = float(entry['exists_target'])

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask

        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        target_ids.masked_fill_(exists_target.unsqueeze(1) == 0, -100)

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['input_texts'] = input_texts
        batch_entry['target_texts'] = target_texts

        batch_entry['exists_target'] = exists_target
        batch_entry['targets'] = torch.tensor(targets, dtype=torch.long)
        batch_entry['all_targets'] = all_targets
        batch_entry['all_target_ids'] = all_target_ids
        # batch_entry['ious'] = ious

        batch_entry['task'] = 'refcoco'
        batch_entry['img_ids'] = img_ids

        # batch_entry['args'] = args

        return batch_entry

class ShapeButtonFineTuneDataset(Dataset):
    def __init__(self, csv_file = None, pkl_file=None, img_file=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.buttons_frame = pd.read_csv(csv_file)
        self.pkl_frame = pd.read_pickle(pkl_file)
        self.img_file = img_file
        self.topk = topk
        self.verbose = verbose
        self.args = args
        self.mode = mode
        self.rank = rank
        self.n_gpus = torch.cuda.device_count()
        self.img_width = 1920
        self.img_height = 1080
        
        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(args.backbone)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(args.backbone)
        else:
            print('not using bart, please check args')
            return
        
    def __len__(self):
        return len(self.buttons_frame)
            
    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args

        datum = self.buttons_frame.iloc[idx, 0:]
        # print("Here is datum", datum)
        
        if self.args.use_vision:
            img_id = datum[0]
            out_dict['img_id'] = img_id
            boxes = self.pkl_frame[str(img_id)+'.png'].pred_boxes.tensor.detach().cpu()
            np_boxes = np.array(boxes)
            # np_boxes[:, [1, 2]] = np_boxes[:, [2, 1]]
            # print("This is np_boxes:", np_boxes)

            input_txt = str(datum[9])
            if(input_txt == 'press square button'):
                ref_box = np.array([datum[1: 5]], dtype=np.float64)
            elif(input_txt == 'press circle button'):
                ref_box = np.array([datum[5: 9]], dtype=np.float64)
            else:
                print('something is wrong')

            ref_box[:, [1, 2]] = ref_box[:, [2, 1]]
            # print("This is ref_box:", ref_box)

            if self.args.shuffle_boxes and self.mode == 'train':
                box_indices = np.arange(len(np_boxes))
                np.random.shuffle(box_indices)

                np_boxes = np_boxes[box_indices]
                # print(np_boxes)

            n_boxes = len(np_boxes)
            out_dict['n_boxes'] = n_boxes

            # ref_reg_box = norm_to_reg(ref_box, self.img_width, self.img_height)
            # print(type(ref_reg_box))
            
            ious = get_iou(
                torch.tensor(np_boxes, dtype=torch.float),
                torch.tensor(ref_box, dtype=torch.float))

            # print(ious)

            threshold = 0.6
            scores = ious.detach().numpy().flatten()
            scores[scores < threshold] = 0
            scores = scores.astype(np.float64)

            exists_target = scores.sum() > 0

            # print(scores)

            if exists_target:
                correct_indices = np.nonzero(scores)[0].tolist()
                prob = scores / scores.sum()
                # print("Here is prob", prob)

                choice = np.random.multinomial(1, prob)
                # print("Here is np.random.multinomial(1, prob)", choice)
                choice = choice.argmax()
                # print("Here is choice", choice)
            else:
                correct_indices = []
                choice = -100

            # Normalize the boxes (to 0 ~ 1)
            np_boxes[:, (0, 2)] /= self.img_width
            np_boxes[:, (1, 3)] /= self.img_height

            boxes = torch.from_numpy(np_boxes)

            boxes.clamp_(min=0.0, max=1.0)

            feats = self.pkl_frame[str(img_id)+'.png'].feats
            
            if self.args.shuffle_boxes and self.mode == 'train':
                feats = feats[box_indices]

            out_dict['boxes'] = boxes
            out_dict['vis_feats'] = feats

        sent = str(datum[9])

        prefix = "visual grounding:"

        input_text = f'{prefix} {sent}'

        if exists_target:
            if self.args.vis_pointer:
                all_target_ids = correct_indices
                target_text = ''
            else:
                target_text = f'<vis_extra_id_{choice}>'
                all_target_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{idx}>' for idx in correct_indices])

        else:
            if self.args.vis_pointer:
                all_target_ids = []
                target_text = ''
            else:
                target_text = ''
                all_target_ids = []

        out_dict['exists_target'] = exists_target
        out_dict['iou'] = ious
        out_dict['target'] = choice
        out_dict['all_targets'] = correct_indices
        out_dict['all_target_ids'] = all_target_ids

        input_ids = self.tokenizer.encode(input_text, max_length=self.args.max_text_length, truncation=True)
        target_ids = self.tokenizer.encode(target_text, max_length=self.args.max_text_length, truncation=True)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['input_text'] = input_text
        out_dict['target_text'] = target_text

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        args = self.args

        B = len(batch)

        if args.use_vision:
            V_L = max([b['n_boxes'] for b in batch])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        exists_target = torch.zeros(B, dtype=torch.float)

        input_texts = []
        target_texts = []
        targets = []
        all_targets = []
        all_target_ids = []
        ious = []
        img_ids = []

        for i, entry in enumerate(batch):

            if args.use_vision:
                boxes[i, :entry['n_boxes']] += entry['boxes']
                vis_feats[i, :entry['n_boxes']] += entry['vis_feats']

                vis_attention_mask[i, :entry['n_boxes']] = 1

            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            input_texts.append(entry['input_text'])
            target_texts.append(entry['target_text'])
            targets.append(entry['target'])
            all_targets.append(entry['all_targets'])
            all_target_ids.append(entry['all_target_ids'])
            # ious.append(entry['iou'])
            img_ids.append(entry['img_id'])


            exists_target[i] = float(entry['exists_target'])

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask

        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        target_ids.masked_fill_(exists_target.unsqueeze(1) == 0, -100)

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['input_texts'] = input_texts
        batch_entry['target_texts'] = target_texts

        batch_entry['exists_target'] = exists_target
        batch_entry['targets'] = torch.tensor(targets, dtype=torch.long)
        batch_entry['all_targets'] = all_targets
        batch_entry['all_target_ids'] = all_target_ids
        # batch_entry['ious'] = ious

        batch_entry['task'] = 'refcoco'
        batch_entry['img_ids'] = img_ids

        # batch_entry['args'] = args

        return batch_entry        
    

def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = RefCOCOFineTuneDataset(
        split,
        # raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed and mode == 'train':
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    loader.task = 'refcoco'

    return loader

def get_button_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):


    verbose = (gpu == 0)

    if mode == 'train':
        shape_img_GT = shape_train_img_GT
        shape_img_features = shape_train_img_features
        shape_img_dir = shape_train_img_dir
    elif mode == 'val':
        shape_img_GT = shape_val_img_GT
        shape_img_features = shape_val_img_features
        shape_img_dir = shape_val_img_dir
    elif mode == 'test':
        shape_img_GT = shape_test_img_GT
        shape_img_features = shape_test_img_features
        shape_img_dir = shape_test_img_dir

    dataset = ShapeButtonFineTuneDataset(
                                        csv_file=shape_img_GT, 
                                        pkl_file=shape_img_features,
                                        img_file=shape_img_dir, 
                                        rank=gpu, 
                                        topk=topk, 
                                        verbose=verbose, 
                                        args=args, 
                                        mode=mode)

    if distributed and mode == 'train':
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler, collate_fn=dataset.collate_fn, drop_last=True)
    elif mode == 'val':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler, collate_fn=dataset.collate_fn, drop_last=True)
    elif mode == 'test':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler, collate_fn=dataset.collate_fn, drop_last=True)
 


    return loader


