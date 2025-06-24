import gc
import glob
import json
import logging
import os.path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
# pil to tensor
to_tensor = transforms.ToTensor()


class MIMICLoader(Dataset):
    def __init__(self, dirname, chunks=None, unchanged_labels=False):
        assert os.path.exists(dirname), '{} does not exist'.format(dirname)
        if os.path.isdir(dirname):
            logging.debug('searching for files in {}'.format(dirname))
            self.chunks = glob.glob(os.path.join(dirname, '*.pkl'))
            logging.debug('found {} chunks'.format(len(self.chunks)))

        else:
            logging.debug('single chunk {}'.format(dirname))
            self.chunks = [dirname]

        assert len(self.chunks) > 0, 'No .pkl files found in {}'.format(dirname)
        self.chunks.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        if chunks is not None:
            assert chunks < len(self.chunks), '{} exceeds number of chunks'.format(chunks)
            self.chunks = self.chunks[:chunks]

        self.unchanged_labels = unchanged_labels
        self.len = 0
        with open(os.path.join(dirname, 'data_length.json'), 'r') as f:
            cache = json.load(f)
            for chunk in self.chunks:
                self.len += cache[os.path.basename(chunk)]

        logging.debug('total number of images: {}'.format(self.len))

        self.data = {}
        self.current_chunk = 0
        self.offset = 0
        self.limit = 0

    def free_data(self):
        # free memory to load next chunk
        self.data = None
        gc.collect()

    def load_chunk(self, index):
        assert 0 <= index <= len(self.chunks), 'index out of range'
        logging.debug('loading chunk {}'.format(index))
        with open(self.chunks[index], 'rb') as f:
            if index == 0:
                # reset chunks
                self.current_chunk = 0
                self.offset = 0
                self.free_data()
                self.data = pickle.load(f)
                self.limit = len(self.data['image_name'])

            else:
                assert index == self.current_chunk + 1, 'chunks must be loaded in order'
                # loading next chunk
                self.current_chunk = index
                self.offset += len(self.data['image_name'])
                self.free_data()
                self.data = pickle.load(f)
                self.limit += len(self.data['image_name'])

        logging.debug(f'limit {self.limit}, offset {self.offset}, current chunk {self.current_chunk}')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if index == 0:
            self.free_data()
            self.load_chunk(0)

        elif index >= self.limit:
            if self.current_chunk == len(self.chunks) - 1:
                # last chunk, reset iteration
                self.load_chunk(0)

            else:
                # load next chunk
                self.load_chunk(self.current_chunk+1)

        payload = {}
        for key in self.data.keys():
            payload[key] = self.data[key][index-self.offset]
            if key == 'image_tensor':
                payload[key] = to_tensor(payload[key])

        return payload

    def get_loader(self, batch_size):
        indices = np.arange(self.len)
        sampler = torch.utils.data.SequentialSampler(indices)
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           shuffle=False,
                                           collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        data = {}
        for e in batch:
            for key in e.keys():
                if key not in data.keys():
                    data[key] = []

                data[key].append(e[key])

        new_labels = {}
        if not self.unchanged_labels:
            # organize labels for classification training
            for label in data['labels']:
                for key in label:
                    if key not in new_labels.keys():
                        new_labels[key] = []

                    new_labels[key].append(label[key])

            # list to tensor
            for key in new_labels.keys():
                new_labels[key] = torch.tensor(new_labels[key]).to(dtype=torch.long)

            data['labels'] = new_labels

        if 'image_embeddings' in data.keys():
            # loaded a pkl with embeddings
            data['image_embeddings'] = torch.stack(data['image_embeddings'])

        if 'text_embeddings' in data.keys():
            data['text_embeddings'] = torch.stack(data['text_embeddings'])

        if 'image_tensor' in data.keys():
            data['image_tensor'] = torch.stack(data['image_tensor'])

        return data
