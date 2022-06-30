import torch
import torch.utils.data as data
import os
import numpy as np
from PIL import Image
from build_vocab import Vocabulary
from utility import clean
from pycocotools.coco import COCO

class CocoDataset(data.Dataset):

    def __init__(self, args, vocab, transform):

        self.root = args.image_dir
        self.coco = COCO(args.caption_path)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.rcnn = (args.cnn == 'rcnn')
        self.truncate = args.truncate + 1

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        tokens = clean(coco.anns[ann_id]['caption'])
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        if self.rcnn:
            image = np.load(os.path.join(self.root, str(img_id)+'.npz'))['feat']
            # np.random.shuffle(image)
            image = torch.from_numpy(image)
        else:
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            image = self.transform(image)
        caption = [vocab('<start>')]
        caption.extend([vocab(token) for token in tokens])
        if self.truncate > 0:
            if len(caption) >= self.truncate:
                caption = caption[:self.truncate]
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) - 1 for cap in captions]
    targets = torch.zeros(len(captions), max(lengths) + 1).long()
    for i, cap in enumerate(captions):
        end = lengths[i] + 1
        targets[i, :end] = cap[:end]        
    return images, targets, lengths, None

class CocoDataset_RL(data.Dataset):

    def __init__(self, args, vocab, transform):

        self.root = args.image_dir
        coco = COCO(args.caption_path)
        self.ids = list(coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        gts = {}
        for i in range(len(self.ids)):
            ann_id = self.ids[i]
            img_id = coco.anns[ann_id]['image_id']
            if img_id in gts.keys():
                gts[img_id].append(' '.join(clean(coco.anns[ann_id]['caption'])))
            else:
                gts[img_id] = [' '.join(clean(coco.anns[ann_id]['caption']))]
        self.gts = gts
        self.coco = coco
        self.rcnn = (args.cnn == 'rcnn')
        self.truncate = args.truncate + 1

    def __getitem__(self, index):

        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        tokens = clean(coco.anns[ann_id]['caption'])
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        if self.rcnn:
            image = torch.from_numpy(np.load(os.path.join(self.root, str(img_id)+'.npz'))['feat'])
        else:
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            image = self.transform(image)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        if self.truncate > 0:
            if len(caption) >= self.truncate:
                caption = caption[:self.truncate]
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, self.gts[img_id]

    def __len__(self):
        return len(self.ids)

def collate_fn_RL(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, gts = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) - 1 for cap in captions]
    targets = torch.zeros(len(captions), max(lengths) + 1).long()
    for i, cap in enumerate(captions):
        end = lengths[i] + 1
        targets[i, :end] = cap[:end]
    return images, targets, lengths, gts

class F30kDataset(data.Dataset):

    def __init__(self, args, vocab, transform=None):

        self.root = args.image_dir
        import json
        with open(args.caption_path, 'r') as f:
            dataset = json.load(f)
        self.dataset = dataset['images']
        self.vocab = vocab
        self.transform = transform
        self.truncate = args.truncate + 1

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        vocab = self.vocab
        tokens = clean(self.dataset[index]['caption'])
        img_id = self.dataset[index]['image_id']
        path = img_id + '.jpg'
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        if len(caption) >= self.truncate:
            caption = caption[:self.truncate]
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.dataset)


def get_data_loader(args, vocab, transform):
    if args.dataset == 'coco':
        if args.SCST:
            coco = CocoDataset_RL(args, vocab, transform)
            data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.num_workers, collate_fn=collate_fn_RL)
        else:
            coco = CocoDataset(args, vocab, transform)
            data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        f30k = F30kDataset(args, vocab, transform)
        data_loader = torch.utils.data.DataLoader(dataset=f30k, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, collate_fn=collate_fn)
    return data_loader


class CocoDataset_test(data.Dataset):

    def __init__(self, args, transform):
        self.root = args.image_path
        self.image_size = args.image_size
        self.coco = COCO(args.caption_path)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.rcnn = (args.cnn == 'rcnn')

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        path = coco.imgs[img_id]['file_name']
        if self.rcnn:
            image = np.load(os.path.join(self.root, str(img_id)+'.npz'))['feat']
            image = torch.from_numpy(image)
        else:
            image = Image.open(os.path.join(self.root, path)).convert('RGB')
            image = image.resize([self.image_size, self.image_size], Image.LANCZOS)
            image = self.transform(image)
        return image, img_id

    def __len__(self):
        return len(self.ids)


# Instanced by batch_test.py for test acceleration. Only work for coco for now
def get_test_data_loader(args, transform):
    if args.dataset == 'coco':
        coco = CocoDataset_test(args, transform)
        data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=args.batch_size, num_workers=2)
    else:
        data_loader = None
    return data_loader