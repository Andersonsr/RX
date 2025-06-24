import glob
import os.path
import pickle
import math
from tqdm import tqdm
from torchvision import transforms
import logging
import time
import gc
import json
from model.classifiers import mimic_classifier_list
# pil to tensor
to_tensor = transforms.ToTensor()


def chunks_pil_to_tensor(dirname):
    assert False, 'dont use me, chunks got 5 times larger'
    assert os.path.exists(dirname), '{} does not exist'.format(dirname)
    chunks = glob.glob(dirname+'/chunk*.pkl')
    assert len(chunks) > 0, 'No chunks found at {}'.format(dirname)
    for i, chunk in enumerate(chunks):
        print('chunk {}/{}'.format(i, len(chunks)))
        tensors = []
        with open(chunk, 'rb') as f:
            chunk_data = pickle.load(f)
            for image in tqdm(chunk_data['image_tensor']):
                tensors.append(to_tensor(image))

            chunk_data['image_tensor'] = tensors
            new_data = chunk_data

        with open(chunk, 'wb') as f:
            pickle.dump(new_data, f)


def mimic_chunk_labels(filename):
    '''
    edit chunk file with reorganized labels
    :param filename: chunk pkl file to edit
    :return: None
    '''
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        for i, e in enumerate(data['labels']):
            # old labels: positive: 1, negative: 0, uncertain: -1, ignore: nan
            # new labels: positive: 1, negative: 0, uncertain: 2, ignore: 3
            new_labels = {}
            for label in mimic_classifier_list:
                if label not in e.keys():
                    new_labels[label] = 3
                    print('image {} missing label: {}'.format(i, label))
                else:
                    if math.isnan(e[label]):
                        new_labels[label] = 3
                    else:
                        new_labels[label] = 2 if e[label] < 0 else e[label]
            data['labels'][i] = new_labels

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def count_length(dirname):
    total_len = 0
    cache = {}
    assert os.path.exists(dirname), '{} does not exist'.format(dirname)
    chunks = glob.glob(os.path.join(dirname, "chunk*.pkl"))
    assert len(chunks) > 0, "No chunks found at {}".format(dirname)
    for chunk in tqdm(chunks):
        logging.debug('loading chunk {} ...'.format(chunk))
        starting_time = time.time()
        with open(chunk, 'rb') as f:
            data = None
            gc.collect()
            data = pickle.load(f)
            ending_time = time.time()
            logging.debug('load time: {}'.format(ending_time - starting_time))
            length = len(data['image_name'])

        logging.info('{} length is {}'.format(chunk, length))
        cache[os.path.basename(chunk)] = length
        total_len += length

    with open(os.path.join(dirname, 'data_length.json'), 'w') as f:
        json.dump(cache, f)


if __name__ == '__main__':
    chunks =glob.glob('E:\\datasets\\mimic\\mimic_train_256\\chunks\\chunk*.pkl')
    for chunk in tqdm(chunks):
        mimic_chunk_labels(chunk)
    # count_length('E:\\datasets\\mimic\\mimic_dev_224\\chunks')

