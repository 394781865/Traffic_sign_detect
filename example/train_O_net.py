import argparse
import sys
sys.path.append('..')
from core.imdb import IMDB
from train import train_net
from core.model import O_Net

data_name='wider'
model_path='../data/%s_model/onet'%data_name

def train_O_net(image_set, root_path, dataset_path, prefix,
                end_epoch, frequent, lr):
    imdb = IMDB(data_name, image_set, root_path, dataset_path)
    gt_imdb = imdb.gt_imdb()
    gt_imdb = imdb.append_flipped_images(gt_imdb)
    sym = O_Net

    train_net(sym, prefix,   end_epoch, gt_imdb,
              48, frequent, lr)

def parse_args():
    parser = argparse.ArgumentParser(description='Train 48-net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_set', dest='image_set', help='training set',
                        default='train_48', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='../data', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='../data/%s'%data_name, type=str)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=model_path, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=16, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=0.01, type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    train_O_net(args.image_set, args.root_path, args.dataset_path, args.prefix,
                args.end_epoch, args.frequent, args.lr)
