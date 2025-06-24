import json

import torch
from open_clip import create_model_from_pretrained
import torch.nn as nn
import timm


def create_model(config):
    if 'encoder_config' in config.keys():
        # finetuned model
        encoder_config = config['encoder_config']
        ck = torch.load(config['checkpoint_path'])

        if encoder_config['package'] == 'timm':
            model = timm.create_model(config['model_name'], pretrained=True, num_classes=0)
        elif encoder_config['package'] == 'huggingface':
            model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP2-256')
            model = model.visual
        else:
            raise NotImplementedError

        if config['lora']:
            # TODO: apply LoRA
            raise NotImplementedError

        model.load_state_dict(ck['model_state_dict'])
        return model

    # loading pretrained model
    if config['package'] == 'timm':
        return timm.create_model(config['model_name'], pretrained=True, num_classes=0)
    if config['package'] == 'huggingface':
        model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP2-256')
        return model.visual


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.vision = create_model(config)
        if 'encoder_config' in config.keys():
            config = config['encoder_config']

        self.input_size = config['input_size']
        self.output_dim = config['output_dim']

    def forward(self, x):
        return self.vision(x)

    def grid_features(self, x):
        if type(self.vision) is timm.models.vision_transformer.VisionTransformer:
            return self.vision(x, output_hidden_states=True)


if __name__ == '__main__':
    siglip_512 = {'input_size': 512,
              'output_dim': 768,
              'model_name': 'vit_base_patch16_siglip_512',
              'package': 'timm'}

    siglip2_256 = {'input_size': 256,
                   'output_dim': 768,
                   'model_name': 'hf-hub:timm/ViT-B-16-SigLIP2-256',
                   'package': 'huggingface'}

    vision = Encoder(siglip2_256)
    ck = torch.load('D:\\modelos_v2\\encoder\\class3_siglip256\\backbone_checkpoint.pt')
    vision.load_state_dict(ck['model_state_dict'])
    print(vision)

    # json.dump(siglip2_256, open('src/configs/base_siglip2_256.json', 'w'), indent=2)
