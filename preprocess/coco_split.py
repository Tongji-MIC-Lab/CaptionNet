import json
import nltk
from random import shuffle, seed
import argparse
import os

def split():
    seed(123)
    if args.split_option == 'karpathy':
        num_val = 5000
        num_test = 5000
    else:
        num_val = 2000
        num_test = 0
    val = json.load(open(args.val_path, 'r'))
    train = json.load(open(args.train_path, 'r'))

    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']
    shuffle(imgs)

    dataset = {}
    dataset['val'] = imgs[:num_val]
    dataset['test'] = imgs[num_val: num_val + num_test]
    dataset['train'] = imgs[num_val + num_test:]

    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if not imgid in itoa: itoa[imgid] = []
        itoa[imgid].append(a)

    json_data = {}
    info = train['info']
    licenses = train['licenses']
    split = ['val', 'test', 'train']

    for subset in split:
        json_data[subset] = {'type': 'caption', 'info': info, 'licenses': licenses,
                             'images': [], 'annotations': []}
        for img in dataset[subset]:
            img_id = img['id']
            anns = itoa[img_id]
            json_data[subset]['images'].append(img)
            json_data[subset]['annotations'].extend(anns)
        ### tokenize training annotations for training convenience
        if subset == 'train':
            for j in range(len(json_data[subset]['annotations'])):
                json_data[subset]['annotations'][j]['caption'] = nltk.word_tokenize(json_data[subset]['annotations'][j]['caption'])
        json.dump(json_data[subset], open(os.path.join(args.out_dir, args.split_option + '_split_' + subset + '.json'), 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_path', type=str, default='../data/annotations/captions_val2014.json')
    parser.add_argument('--train_path', type=str, default='../data/annotations/captions_train2014.json')
    parser.add_argument('--split_option', type=str, default='karpathy', help='[online, karpathy]')
    parser.add_argument('--out_dir', type=str, default='../data/annotations/')
    args = parser.parse_args()
    assert args.split_option in ['online', 'karpathy']
    split()