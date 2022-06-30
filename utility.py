import argparse
import torch
from cider.pyciderevalcap.ciderD.ciderD import CiderD
from pycocoevalcap.bleu.bleu import Bleu
from collections import OrderedDict
import numpy as np
import torch.nn as nn
import math
import os
from torchvision import transforms

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class self_critical_scorer():
    def __init__(self, df='coco-train-words', bleu4_weight=0.0):
        assert bleu4_weight >= 0 and bleu4_weight <= 1
        self.scorer = CiderD(df=df)
        if bleu4_weight > 0:
            self.bleu_scorer = Bleu(4)
        self.bleu4_weight = bleu4_weight

    def get_reward(self, gts, gen_result, greedy_res, vocab):
        batch_size = gen_result.size(0)
        res = OrderedDict()
        gen_result = gen_result.data.cpu().numpy()
        greedy_res = greedy_res.data.cpu().numpy()

        for i in range(batch_size):
            res[i] = [vocab.tensor2str(gen_result[i])]
        for i in range(batch_size):
            res[batch_size + i] = [vocab.tensor2str(greedy_res[i])]

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
        res__ = {i: res[i] for i in range(2 * batch_size)}
        gts_ = {i: gts[i % batch_size] for i in range(2 * batch_size)}
        if self.bleu4_weight < 1:
            _, cider_scores = self.scorer.compute_score(gts_, res_)
            print('Cider scores:', _)
            cider_scores = cider_scores[:batch_size] - cider_scores[batch_size:]
        else:
            cider_scores = np.zeros(batch_size)
        if self.bleu4_weight > 0:
            _, bleu_scores = self.bleu_scorer.compute_score(gts_, res__)
            bleu_scores = np.array(bleu_scores[3])
            print('BLEU@4 scores:', _[3])
            bleu_scores = bleu_scores[:batch_size] - bleu_scores[batch_size:]
        else:
            bleu_scores = np.zeros(batch_size)
        scores = (1 - self.bleu4_weight) * cider_scores + self.bleu4_weight * bleu_scores
        rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
        return rewards

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class SCST_loss(nn.Module):

    def __init__(self):
        super(SCST_loss, self).__init__()

    def forward(self, input, seq, reward):

        input = to_contiguous(input.view(-1))
        reward = to_contiguous(reward.view(-1))

        mask = to_contiguous((seq>0).float().cuda()).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def isStopWord(word):
    nltk_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                       "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                       "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                       'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
                       'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                       'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                       'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                       'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                       'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                       'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                       'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                       'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                       'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                       "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                       'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                       "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    my_stop_words = ['<start>', '<unk>', "'s", "'t", "n't", '<pad>', '<end>']
    if word in nltk_stop_words or word in my_stop_words:
        return True
    return False

def is_letter(ch):
    if (ord(ch) >= ord('a') and ord(ch) <= ord('z')) or \
            (ord(ch) >= ord('A') and ord(ch) <= ord('Z')) or ch == '\'':
        return True
    return False


def cleanstring(string):
    result = ""
    string = string.lower()
    for ch in string:
        if is_letter(ch):
            result += ch
        else:
            result += ' '
    return result


def clean(caption):
    tmp = []
    assert type(caption) == type([]), "training captions must be tokenized to a list"
    for t in caption:
        tmp.extend(cleanstring(t).split())
    return tmp


class log():
    def __init__(self, eval=False):
        self.log_file = 'train_log' if not eval else 'test_log'
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    
    def print_log(self, info_str, end='\n'):
        print(info_str, end=end)
        with open(self.log_file, 'a') as f:
            f.write(info_str + end)


def train_arg_early_check(args):
    assert args.dataset in ['coco', 'flickr30k'], "dataset must be one of coco and flickr30k"
    assert args.cnn in ['resnet', 'rcnn'], "cnn must be one of resnet and rcnn"
    if not args.other_devices:
        args.other_devices = []
    if args.cnn == 'resnet':
        assert args.resnet in ['101', '152'], "resnet must be one of 101 and 152"
    if args.cnn_fine_tune:
        assert args.cnn == 'resnet', "only resnet can be fine-tuned"

def test_arg_early_check(args):
    assert args.beam_size >= 1
    if args.cnn == 'resnet':
        assert args.resnet in ['101', '152'], "resnet must be one of 101 and 152"
    assert args.dataset in ['coco', 'flickr30k'], "dataset must be one of coco and flickr30k"

def compute_lr(epoch, lr_base, lr_decay_start, lr_decay_every, lr_decay_rate):
    if epoch > lr_decay_start:
        frac = int((epoch - lr_decay_start) / lr_decay_every)
        decay_factor = math.pow(lr_decay_rate, frac)
        return lr_base * decay_factor
    return lr_base

def init_parameter(linear):
    init_range = 0.1
    linear.weight.data.uniform_(-init_range, init_range)
    linear.bias.data.fill_(0)

def clip_gradient(optimizer, value):
    if value <= 0:
        return
    for group in optimizer.param_groups:
        for param in group['params']:
            nn.utils.clip_grad_value_(param, value)

def get_bad_ending_id(vocab):
    bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to',
                   'is', 'are', 'am']
    return [vocab.word2idx[word] for word in bad_endings]

def get_special_tag_id_without_end(vocab):
    return [vocab.word2idx['<start>'], vocab.word2idx['<pad>'], vocab.word2idx['<unk>']]

def get_transform(args, test=False):
    if not test:
        return transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

def get_relating_word_positions(word, vocab):
    positions = [vocab.word2idx[word]]
    special_words = {'child':'children', 'children':'child', 'foot':'feet',
                     'feet':'foot', 'tooth':'teeth', 'teeth':'tooth',
                     'woman':'women', 'women':'woman', 'man':'men', 'men':'man'}
    if word in special_words.keys():
        positions.append(vocab.word2idx[special_words[word]])
    if word[-1] == 'y' and word[:-1] + 'ies' in vocab.word2idx:
        positions.append(vocab.word2idx[word[:-1] + 'ies'])
    if word[-1] in ['o', 's', 'x'] and word + 'es' in vocab.word2idx:
        positions.append(vocab.word2idx[word + 'es'])
    if len(word) > 1 and word[-2:] in ['sh', 'ch'] and word + 'es' in vocab.word2idx:
        positions.append(vocab.word2idx[word + 'es'])
    if word + 's' in vocab.word2idx:
        positions.append(vocab.word2idx[word + 's'])
    if word[-1] == 's' and word[:-1] in vocab.word2idx:
        positions.append(vocab.word2idx[word[:-1]])
    if len(word) > 1 and word[-2:] == 'es' and word[:-2] in vocab.word2idx:
        positions.append(vocab.word2idx[word[:-2]])
    if len(word) > 2 and word[-3:] == 'ies' and word[:-3] + 'y' in vocab.word2idx:
        positions.append(vocab.word2idx[word[:-3] + 'y'])
    return positions
