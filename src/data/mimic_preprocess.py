import os.path
import math
import pickle

from PIL import Image
from tqdm import tqdm
import json


def preprocess(img, crop, size):
    if not crop:
        width, height = img.size
        if height > width:
            new_height = size
            new_width = width * new_height / height
        else:
            new_width = size
            new_height = height * new_width / width
        return img.resize((int(new_width), int(new_height)))


if __name__ == '__main__':
    filename = 'E:\\datasets\\mimic\\chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json'
    output_dir = 'D:\\mimic\\preprocess\\resize'
    root = 'E:\\datasets\\mimic\\mimic-cxr-jpg\\2.1.0\\files\\'
    size = 1024
    crop = False
    chunk_size = 2000
    chunk_counter = 0
    image_counter = 0

    file = json.load(open(filename, 'r'))
    data = {'image_ids': [], 'findings': [], 'image_tensors': [], 'labels': []}
    for i, sample in tqdm(enumerate(file), total=len(file)):
        # print('old', sample['chexpert_labels'])
        new_labels = {}
        if sample['generate_method'] == 'gpt4':
            path = os.path.join(root, sample['image'].replace('mimic/', ''))
            # some images were not correctly downloaded and wget does not work anymore
            if os.path.exists(path):
                image_counter += 1
                im = Image.open(path).convert('RGB')

                for key, item in sample['chexpert_labels'].items():
                    if math.isnan(item):
                        new_labels[key] = 3
                    else:
                        new_labels[key] = 2 if item < 0 else item

                data['image_tensors'].append(preprocess(im, crop, size))
                data['image_ids'].append(sample['id'])
                data['findings'].append(sample['conversations'][1]['value'].replace('\n', ''))
                data['labels'].append(new_labels)

        if (image_counter + 1) % chunk_size == 0 or i == len(file) - 1:
            with open(os.path.join(output_dir, f'chunk_{chunk_counter}.pkl'), 'wb') as chunk_file:
                pickle.dump(data, chunk_file)
                data = {'image_ids': [], 'findings': [], 'image_tensors': [], 'labels': []}

    # save
    with open(os.path.join(output_dir, 'info.json'), 'w') as outfile:
        dict_data = {'total_images': image_counter, 'total_chunks': chunk_counter, 'crop': crop, 'size': size}
        json.dump(dict_data, outfile)

