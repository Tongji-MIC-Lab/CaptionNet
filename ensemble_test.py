
from utility import str2bool, get_transform
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
from Models import *
from PIL import Image
import json
from collections import OrderedDict
import numpy as np
import time

def load_image(image_path, size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([size, size], Image.LANCZOS)
    transform = get_transform(None, test=True)
    image = transform(image).unsqueeze(0)
    return image

def init(args, index, vocab):
    args.crop_size = args.image_size
    args.vocab_size = len(vocab)
    model = load_model(args, device, test=True)
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.save_name + str(index)),
                        map_location={'cuda:%d' % args.model_device[index - 1]: 'cuda:%d' % (args.device)}))
    return model

def evaluation(args):
    captions = []
    with open(args.caption_path) as f:
        data = json.load(f)
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    models = []
    for i in range(args.num_models):
        models.append(init(args, i + 1, vocab))
    ensemble_model = CAPEnsemble(models)
    if args.dataset == 'flickr30k':
        tmp = {}
        tmp['images'] = []
        ids = []
        for item in data['images']:
            if item['image_id'] not in ids:
                tmp['images'].append(item)
                ids.append(item['image_id'])
        data = tmp
    time_start = time.time()
    for counter, d in enumerate(data['images']):
        dict = {}
        if args.dataset == 'coco' and args.cnn == 'resnet':
            image = load_image(os.path.join(args.image_path, d['file_name']), args.image_size).to(device)
            dict['image_id'] = d['id']
        elif args.dataset == 'coco' and args.cnn == 'rcnn':
            image = torch.from_numpy(np.load(os.path.join(args.image_path, str(d['id'])+'.npz'))['feat']).to(device)
            image = image.unsqueeze(0)
            dict['image_id'] = d['id']
        else:
            image = load_image(os.path.join(args.image_path, d['image_id']+'.jpg'), args.image_size).to(device)
            dict['image_id'] = d['image_id']
        with torch.no_grad():
            caption = ensemble_model.beam_search(image, vocab, args.beam_size)
        dict['caption'] = caption
        print(counter, "  ", dict)
        captions.append(dict)
    path = os.path.join(
        args.result_path,
        '%s_b%d_captions_val2014_fakecap_results.json'%
        (args.result_name,args.beam_size))
    with open(path,'w') as f:
        json.dump(captions, f)
    print('Total testing time: %.2f min' % ((time.time() - time_start) / 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_models', type=int, default=4, help='number of models to ensemble')
    parser.add_argument('--image_path', type=str, default='png/4.jpg', help='input image for generating caption')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--model_path', type=str, default='models/',
                        help='directory containing trained models')
    parser.add_argument('--save_name', type=str, default='CAP')
    parser.add_argument('--model_name', type=str, default='CAP')
    parser.add_argument('--caption_path', type=str, default='data/annotations/karpathy_split_test.json',
                        help='caption test split path for test')
    parser.add_argument('--result_name', type=str, default='/home/yly/python3/bin/results',
                        help='caption result for test')
    parser.add_argument('--result_path', type=str, default='FUN',
                        help='caption path for test')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size for beam search')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_device', nargs='+', type=int)
    parser.add_argument('--cnn', type=str, default='resnet')
    parser.add_argument('--resnet', type=str, default='152')
    parser.add_argument('--multi_gpu_model', type=str2bool, default='True')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--sequential_length', type=int, default=10)
    ### Ablation Studis
    parser.add_argument('--eliminate_code', type=int, default=0,
                        help='10 for eliminating IFE, 01 for eliminating CapNet')

    args = parser.parse_args()
    assert(args.num_models == len(args.model_device))
    device = torch.device('cuda')
    torch.cuda.set_device(args.device)
    evaluation(args)
