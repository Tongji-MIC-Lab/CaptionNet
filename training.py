import utility
import argparse
import os
import pickle
from data_loader import get_data_loader
from build_vocab import Vocabulary
from torch.nn.utils.rnn import pack_padded_sequence
from Models import *
import math
import time

def main(args):
    device = torch.device('cuda')
    torch.cuda.set_device(args.device)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    args.vocab_size = len(vocab)
    data_loader = get_data_loader(args, vocab, utility.get_transform(args))
    model = load_model(args, device)

    if args.load_pre_model:
        try:
            last_epoch = int(args.pre_model.split('-')[1].split('.')[0])
        except:
            last_epoch = 0
        model.load_state_dict(torch.load(os.path.join(
            args.model_path, args.pre_model), map_location={'cuda:%d' % args.pre_device: 'cuda:%d' % (args.device)}),
            strict=False
        )
    else:
        last_epoch = 0

    if args.cnn_fine_tune:
        cnn_subs = list(model.module.encoder.CNN.children())[args.fine_tune_start_layer:]
        cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
        cnn_params = [item for sublist in cnn_params for item in sublist]
        cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.lr_cnn)

    params = model.module.get_params()
    if args.SCST:
        loss_function = utility.SCST_loss()
        SCST_scorer = utility.self_critical_scorer(df=args.df, bleu4_weight=args.bleu4_weight)
    else:
        loss_function = nn.CrossEntropyLoss()
    model.train()
    total_step = len(data_loader)
    training_time = []
    for epoch in range(args.num_epochs):
        time_start = time.time()
        lr = utility.compute_lr(last_epoch + epoch + 1, args.lr,
                                args.lr_decay, args.lr_decay_every, args.lr_decay_rate)
        if args.cnn_fine_tune:
            log.print_log('With CNN fine tune')
        optimizer = torch.optim.Adam(params, lr=lr)
        log.print_log('Learning Rate for Epoch %d: %.6f' % (last_epoch + epoch + 1, lr))
        total_loss = 0.0
        for i, (images, captions, lengths, gts) in enumerate(data_loader):
            with torch.autograd.set_detect_anomaly(args.detect_anomaly):
                captions, images = captions.to(device), images.to(device)
                targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]
                captions = captions[:, :-1]

                if not args.SCST:
                    outputs = model(images, captions, lengths, args)
                    outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
                    loss = loss_function(outputs, targets)
                else:
                    gen_results, sample_logprobs = model(images, captions, lengths, args, vocab=vocab)
                    model.eval()
                    greedy_res, _ = model(images, captions, lengths, args, vocab=vocab)
                    model.train()
                    reward = torch.from_numpy(SCST_scorer.get_reward(gts, gen_results, greedy_res, vocab)).float().cuda()
                    loss = loss_function(sample_logprobs, gen_results, reward)
                model.zero_grad()
                loss.backward()
                utility.clip_gradient(optimizer, args.clip)
                optimizer.step()
                if args.cnn_fine_tune:
                    cnn_optimizer.step()
                total_loss += loss.item()

                if (i + 1) % args.log_step == 0:
                    log.print_log('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'
                        .format(last_epoch + epoch + 1, last_epoch + args.num_epochs, i + 1, total_step,
                                total_loss / args.log_step), end = ' ')
                    log.print_log('| training time so far: %.3f h' % round((time.time() - time_start) / 3600, 3))
                    total_loss = 0.0
                if (i + 1) % args.save_step == 0:
                    torch.save(model.state_dict(), os.path.join(
                        args.model_path, '%s%d.ckpt' % (args.save_name, i + 1)))

        torch.save(model.state_dict(), os.path.join(
            args.model_path, '%s-%d.ckpt' % (args.save_name, 1 + epoch + last_epoch)))
        time_end = time.time()
        log.print_log('Training time for last epoch: %.3f h' % (int(time_end - time_start) / 3600))
        training_time.append(round((time_end - time_start) / 3600, 3))

    log.print_log('Training finished, time for each epoch: ' + str(training_time))
    log.print_log('Total training time: %.3f h' % ((sum(training_time))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    str2bool = utility.str2bool
    parser.add_argument('--dataset', type=str, default='coco', help='train on which dataset, coco or flickr30k')
    parser.add_argument('--load_pre_model', type=str2bool, default='False', help='load pre_model or not')
    parser.add_argument('--pre_model', type=str, default='CAPB.ckpt', help='if load_pre_model, the pre model name')
    parser.add_argument('--pre_device', type=int, default=3, help='which device was the pre_model trained')
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--save_name', type=str, default='CAP', help='name for saving trained models')
    parser.add_argument('--model_name', type=str, default='ATT', help='model name for training')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/karpathy_split_train_token.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=500, help='step size for saving trained models')
    parser.add_argument('--use_multi_gpu', type=str2bool, default='False', help='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--other_devices', nargs='+', type=int)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay', type=int, default=99, help='epoch at which to start lr decay')
    parser.add_argument('--lr_decay_every', type=int, default=50,
                        help='decay learning rate at every this number')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--truncate', type=int, default=-1, help='truncate sentence length')
    parser.add_argument('--detect_anomaly', type=str2bool, default='f',
                        help='enable autograd anomaly detection or not. If true, training time might increase')

    ### CNN settings
    parser.add_argument('--cnn', type=str, default='resnet', help='cnn kind, resnet or rcnn')
    parser.add_argument('--resnet', type=str, default='152', help='if cnn==resnet, resnet type, 152 or 101')
    parser.add_argument('--cnn_fine_tune', type=str2bool, default='False', help='start fine-tuning CNN or not')
    parser.add_argument('--fine_tune_start_layer', type=int, default=5, help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--lr_cnn', type=float, default=1e-4, help='learning rate for fine-tuning CNN')

    ### LSTM settings
    parser.add_argument('--sequential_length', type=int, default=10)
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--clip', type=float, default=-1, help='clip gradient out of [-clip, clip], if clip <= 0, disable gradient clip')

    ### RL settings
    parser.add_argument('--SCST', type=str2bool, default='f', help='use self-critical training')
    parser.add_argument('--sample_beam_size', type=int, default=1, help='sampling from decoded beam')
    parser.add_argument('--baseline_beam_size', type=int, default=1, help='using beam search result as baseline')
    parser.add_argument('--disable_dropout', type=str2bool, default='f', help='disable dropout in SCST')
    parser.add_argument('--df', type=str, default='coco-train-words',
                        help='file name to store the df for CIDEr-D computing')
    parser.add_argument('--bleu4_weight', type=float, default=0,
                        help='reward weight for BLEU@4. (1 - bleu4_weight) will assign to CIDEr-D reward')

    ### Ablation Studis
    parser.add_argument('--eliminate_code', type=int, default=0,
                        help='2 for eliminating IFE, 1 for eliminating CapNet, and 3 for eliminating both')

    args = parser.parse_args()

    ### Early check for arguments
    utility.train_arg_early_check(args)

    log = utility.log()
    log.print_log(str(args))
    torch.backends.cudnn.benchmark = True
    main(args)
