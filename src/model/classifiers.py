import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_classes):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, output_classes)

    def forward(self, x):
        return self.fc1(x)


class MultiClassifier(nn.Module):
    def __init__(self, classifiers_list, input_size, output_classes):
        super(MultiClassifier, self).__init__()
        self.classifiers_list = classifiers_list
        for classifier in classifiers_list:
            self.add_module(classifier, LinearClassifier(input_size, output_classes))

    def forward(self, x):
        y = {}
        for name, module in self.named_children():
            y[name] = module(x)
        return y


if __name__ == '__main__':
    classification_list = ['Atelectasis',
                           'Cardiomegaly',
                           'Consolidation',
                           'Edema',
                           'Enlarged Cardiomediastinum',
                           'Fracture',
                           'Lung Lesion',
                           'Lung Opacity',
                           'Pleural Effusion',
                           'Pleural Other',
                           'Pneumonia',
                           'Pneumothorax']

    multi = MultiClassifier(classification_list, 768, 3)
    x = torch.rand((1, 768))
    print(x.shape)
    y = multi(x)
    # print(y.shape)
    print(y)

