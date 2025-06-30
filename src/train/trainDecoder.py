import argparse
import json
import logging
import os
import sys
import glob
import torch
from torch.optim import AdamW
from tqdm import tqdm
# path trick
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from data.mimic_dataloader import MIMICLoader
from model.decoder import Decoder
from model.mapper import SimpleMapper
from util import learnable_parameters


def save_checkpoint(state, path, keep=3):
    previous = glob.glob(os.path.join(os.path.dirname(path), 'decoder_*.pt'))
    previous.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]), reverse=True)
    if len(previous) > keep:
        os.remove(previous[-1])
    torch.save(state, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset', choices=['mimic'])
    parser.add_argument('--model_name', type=str, required=True, help='Model name',)
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)
    parser.add_argument('--cpu', action='store_true', help='cpu mode', default=False)
    parser.add_argument('--embeddings', type=str, required=True, help='embeddings dir path')
    parser.add_argument('--stage2', action='store_true', help='apply stage2', default=False)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('-logging_interval', type=int, default=10000, help='')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--lora', action='store_true', help='apply lora', default=False)
    parser.add_argument('--lora_rank', type=int, default=16, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout parameter')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logging.info('model device {}'.format(device))

    if args.dataset == 'mimic':
        train_data = MIMICLoader(args.embeddings)
        val_data = MIMICLoader(args.embeddings.replace('train', 'dev'))
    else:
        raise ValueError(f'{args.dataset} is not a valid dataset')

    train_loader = train_data.get_loader(args.batch_size)
    val_loader = val_data.get_loader(args.batch_size)

    vis_dim = train_data[0]['image_embeddings'].shape[1]

    decoder = Decoder(args.model_name, vis_dim, SimpleMapper)

    optimizer = AdamW(decoder.parameters())

    # frozen decoder, train only the mapper alignment
    if args.stage2:
        decoder.model.eval()
        logging.info(f'stage2 LLM {learnable_parameters(decoder.model)}')
        logging.info(f'mapper {learnable_parameters(decoder.mapper)}')
        for batch in tqdm(train_loader):
            loss = decoder.forward(batch['image_embeddings'], batch['captions'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if args.lora:
        decoder.lora_model(args.lora_rank, args.lora_alpha, args.lora_dropout, )
    else:
        decoder.train()

    log = {'train_loss': [], 'train_steps': [], 'val_loss': [], 'val_steps': []}
    for epoch in range(args.epochs):
        training_loss = []
        logging.info(f'LLM {learnable_parameters(decoder.model)}')

        for i, batch in enumerate(tqdm(train_loader)):
            loss = decoder.forward(batch['image_embeddings'], batch['captions'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.detach().cpu())

            # logging step
            if (i + 1) % args.logging_interval == 0 or i == len(train_loader)-1:
                log['train_loss'].append(sum(training_loss)/len(training_loss))
                log['train_steps'].append(i + epoch*len(train_loader))
                json.dump(log, open(os.path.join(args.output_dir, 'log.json'), 'w'), indent=2)
                training_loss = []

            # validation step
            if (i + 1) % args.validation_interval == 0 or i == len(train_loader) - 1:
                validation_loss = []
                for batch in tqdm(val_loader):
                    with torch.no_grad():
                        loss = decoder(batch['image_embeddings'], batch['captions'])
                        validation_loss.append(loss.detach.cpu())

                log['val_loss'].append(sum(validation_loss)/len(validation_loss))
                log['val_steps'].append(i + epoch * len(train_loader))
                json.dump(log, open(os.path.join(args.output_dir, 'log.json'), 'w'), indent=2)

                model_dict = {'step': log['val_steps'][-1],
                              'model_state_dict': decoder.model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'loss': log['val_steps'][-1]
                              }
                save_checkpoint(model_dict, os.path.join(args.output_dir,
                                                         'checkpoint_{}.pt'.format(log['val_steps'][-1])))


