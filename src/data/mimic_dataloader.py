import json
import os

import torch


class MimicDataloader(torch.utils.data.Dataset):
    def __init__(self, filename):
        super()
        assert os.path.exists(filename), f"No such file or directory: {filename}"
        with open(filename, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self, index):
        return len(self.data)

    def get_loader(self, batch_size):
        sampler = torch.utils.data.SequentialSampler(range(len(self.data)))
        return torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=sampler, shuffle=False)


if __name__ == '__main__':
    dataset = MimicDataloader('E:\\datasets\\mimic\\chat_dev_MIMIC_CXR_all_gpt4extract_rulebased_v1.json')
    loader = dataset.get_loader(8)
    print(dataset[0])

