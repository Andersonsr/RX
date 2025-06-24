import argparse
import json
import logging
import os
import sys
import torch
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from data.dataloaders import MIMICLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset', choices=['mimic'])
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--config', type=str, required=True, help='decoder config file path')
    parser.add_argument('--embeddings', type=str, required=True, help='embeddings dir path')
    parser.add_argument('--stage2', action='store_true', help='apply stage2', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logging.info('model device {}'.format(device))

    if args.dataset == 'mimic':
        train_data = MIMICLoader(args.embeddings)
        val_data = MIMICLoader(args.embeddings.replace('train', 'dev'))




