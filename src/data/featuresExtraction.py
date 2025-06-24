import argparse
import glob
import logging
import os
import pickle
from tqdm import tqdm
import torch
import sys
import numpy as np
import json
path = os.path.normpath(os.path.join(os.path.join(os.path.abspath(__file__)), '..', '..'))
sys.path.append(path)
from torch.utils.data import Dataset
from model.encoder import Encoder
from torchvision import transforms
# pil to tensor
to_tensor = transforms.ToTensor()


class MIMICChunkLoader(Dataset):
    def __init__(self, pkl_file,):
        assert os.path.exists(pkl_file), '{} does not exist'.format(pkl_file)

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            self.id = data['id']
            self.image_name = data['image_name']
            self.image_path = data['image_path']
            self.labels = data['labels']
            self.findings = data['findings']
            self.image_tensor = data['image_tensor']

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        payload = {'image_path': self.image_path[index],
                   'labels': self.labels[index],
                   'findings': self.findings[index],
                   'image': to_tensor(self.image_tensor[index]),
                   'image_name': self.image_name[index],
                   'id': self.id[index]}
        return payload

    # torch dataloader cant handle PIL image by default
    def collate_fn(self, batch):
        data = {'image_path': [], 'labels': [], 'findings': [], 'image_name': [], 'id': [], 'image': []}
        for d in batch:
            data['image_path'].append(d['image_path'])
            data['labels'].append(d['labels'])
            data['findings'].append(d['findings'])
            data['image_name'].append(d['image_name'])
            data['id'].append(d['id'])
            data['image'].append(d['image'])

        data['image'] = torch.stack(data['image'])
        return data

    def get_loader(self, batch_size):
        indices = np.arange(len(self.id))
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False,
                                           collate_fn=self.collate_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str, required=True,
                        help='dir containing chunks')
    parser.add_argument('--output', type=str, required=True, help='dir to save embeddings chunks')
    parser.add_argument('--config', type=str, required=True, help='encoder config',)
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunks = glob.glob(os.path.join(args.dirname, '*.pkl'))

    logger = logging.getLogger('captioning')
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    config = json.load(open(args.config, 'r'))
    model = Encoder(config)

    os.makedirs(args.output, exist_ok=True)

    for chunk in chunks:
        data = {'image_name': [], 'image_id': [], 'image_embeddings': [], 'captions': [],
                'labels': [], 'grid_embeddings': []}

        logging.info('Loading chunk: {}'.format(chunk))
        json_data = MIMICChunkLoader(chunk)
        loader = json_data.get_loader(args.batch_size)
        logging.debug('Loaded chunk len: {}'.format(len(json_data)))

        for batch in tqdm(loader):
            logging.debug('batch size: {}'.format(len(batch['id'])))
            with torch.no_grad():
                logging.debug('image shape {}'.format(batch['image'].shape))
                image_embeddings = model(batch['image']).detach().cpu()
                grid_embeddings = model.grid_features(batch['image']).detach().cpu()
                data['image_embeddings'] += image_embeddings
                data['grid_embeddings'] += grid_embeddings
                logging.debug('batch image embeddings shape: {}'.format(image_embeddings.shape))
                logging.debug('batch grid embeddings shape: {}'.format(grid_embeddings.shape))

        data['image_name'] = json_data.image_name
        data['image_id'] = json_data.id
        data['captions'] = json_data.findings
        data['labels'] = json_data.labels
        data['image_embeddings'] = torch.stack(data['image_embeddings'], dim=0).unsqueeze(dim=1)
        data['grid_embeddings'] = torch.stack(data['grid_embeddings'], dim=0).unsqueeze(dim=1)

        logging.debug('final image embeddings shape: {}'.format(data['image_embeddings'].shape))
        logging.debug('final text embeddings shape: {}'.format(data['text_embeddings'].shape))

        with open(os.path.join(args.output, os.path.basename(chunk)), 'wb') as f:
            pickle.dump(data, f)



