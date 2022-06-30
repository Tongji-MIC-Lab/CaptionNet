# Almost the same with test.py.
# The only difference is using test data loader during evaluation, which can speed up a bit

import argparse
import pickle
import os
from build_vocab import Vocabulary
from data_loader import get_test_data_loader
from Models import *
from PIL import Image
import json
from collections import OrderedDict
import numpy as np
import time


def load_image(image_path, size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([size, size], Image.LANCZOS)
    transform = utility.get_transform(args, test=True)
    image = transform(image).unsqueeze(0)
    return image


def init(args):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    args.vocab_size = len(vocab)
    args.crop_size = args.image_size
    model = load_model(args, device, test=True)
    model.load_state_dict(torch.load(args.model_path,
                        map_location={'cuda:%d' % args.model_device: 'cuda:%d' % (args.device)}))
    return vocab, model


def main(args):
    vocab, model = init(args)
    image = load_image(args.image_path, args.image_size).to(device)
    print(model.module.beam_search(image, vocab, args.beam_size))


def evaluation(args):
    captions = []
    test_data_loader = get_test_data_loader(args, utility.get_transform(args, test=True))
    vocab, model = init(args)
    counter = 1
    '''
    if args.dataset == 'flickr30k':
        tmp = {}
        tmp['images'] = []
        ids = []
        for item in data['images']:
            if item['image_id'] not in ids:
                tmp['images'].append(item)
                ids.append(item['image_id'])
        data = tmp
    '''
    time_start = time.time()
    log = utility.log(eval=True)
    for i, (images, ids) in enumerate(test_data_loader):
        with torch.no_grad():
            images, ids = images.to(device), ids.long()
            v = model.module.encoder(images)
            for b in range(ids.size(0)):
                if args.model_name == 'GLO':
                    result = model.module.beam_search((v[0][0][:, b].unsqueeze(0), v[0][1][:, b].unsqueeze(0)),
                                                     vocab, args.beam_size, test=False)
                elif args.model_name == 'CAP':
                    result, _ = model.module.beam_search(((v[0][0][:, b].unsqueeze(0), v[0][1][:, b].unsqueeze(0)),
                                 v[1][b].unsqueeze(0)), vocab, args.beam_size, test=False)
                elif args.model_name == 'TopDown':
                    result, _ = model.module.beam_search((v[0][b].unsqueeze(0), v[1][b].unsqueeze(0)), vocab,
                                     args.beam_size, test=False)
                dict = {}
                dict['image_id'] = ids[b].item()
                dict['caption'] = vocab.tensor2str(result)
                log.print_log(str(counter) + "  " + str(dict))
                captions.append(dict)
                counter += 1
    path = os.path.join(
        args.result_path, '%s_b%d_captions_val2014_fakecap_results.json' %
        (args.result_name, args.beam_size))
    with open(path, 'w') as f:
        json.dump(captions, f)
    log.print_log('Total testing time: %.2f min' % ((time.time() - time_start) / 60))


def showAll(args):
    vocab, model = init(args)
    for i in range(1, len(os.listdir(args.image_path)) + 1):
        image = load_image(os.path.join(args.image_path, "%d.jpg" % i), args.image_size).to(device)
        print(i, ' ', model.module.beam_search(image, vocab, args.beam_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval', type=int, default=0,
                        help='eval mode, 0 for one image test and 1 for test split evaluation')
    parser.add_argument('--image_path', type=str, default='png/4.jpg', help='input image for generating caption')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--model_path', type=str, default='models/CA1-25.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--model_name', type=str, default='CAP')
    parser.add_argument('--caption_path', type=str, default='data/annotations/karpathy_split_test.json',
                        help='caption test split path for test')
    parser.add_argument('--result_name', type=str, default='CA1', help='caption result name')
    parser.add_argument('--result_path', type=str, default='results',
                        help='path for storing the test results')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size for beam search')
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_device', type=int, default=3)
    parser.add_argument('--cnn', type=str, default='resnet')
    parser.add_argument('--resnet', type=str, default='152')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size for evaluation')
    parser.add_argument('--sequential_length', type=int, default=10)

    ### Ablation Studis
    parser.add_argument('--eliminate_code', type=int, default=0,
                        help='10 for eliminating IFE, 01 for eliminating CapNet')

    args = parser.parse_args()
    ### Early check for arguments
    utility.test_arg_early_check(args)

    device = torch.device('cuda')
    torch.cuda.set_device(args.device)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.eval == 0:
        main(args)
    elif args.eval == 1:
        evaluation(args)
    elif args.eval == 2:
        showAll(args)
