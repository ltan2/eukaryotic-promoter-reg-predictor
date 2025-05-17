import torch
import math
import argparse
import torch.optim as optim
from torch import nn
from icecream import ic
from pathlib import Path

from dataloader import load_data
from deepromoter import DeePromoter
from test import evaluate, mcc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(path, pretrain_w, epoch_num, learn_rate, kernel_size):
    print("TODO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset file.")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained model weights.")
    parser.add_argument("--epoch_num", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate.")
    parser.add_argument("--ker", nargs="+", type=int, default=[27, 14, 7], help="Kernel sizes for CNN layers.")

    args = parser.parse_args()

    # Run the training/testing pipeline
    train_model(args.data_path, args.pretrain, args.epoch_num, args.lr, args.ker)

if __name__ == "__main__":
    main()