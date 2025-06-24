import json
import torch
from open_clip import create_model_from_pretrained
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import timm
import open_clip


def create_model(config):
    if 'encoder_config' in config.keys():
        # finetuned model
        encoder_config = config['encoder_config']
        ck = torch.load(config['checkpoint_path'])

        if encoder_config['package'] == 'timm':
            model = timm.create_model(config['model_name'], pretrained=True, num_classes=0)
        elif encoder_config['package'] == 'openclip':
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
    if config['package'] == 'openclip':
        model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP2-256')
        return model.visual
    if config['package'] == 'transformers':
        return AutoModel.from_pretrained(config['model_name'])


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.vision = create_model(config)
        if 'encoder_config' in config.keys():
            config = config['encoder_config']

        self.input_size = config['input_size']
        self.output_dim = config['output_dim']
        self.package = config['package']

    def forward(self, x):
        if self.package == 'timm' or self.package == 'openclip':
            return self.vision(x)

        elif self.package == 'transformers':
            return self.vision(x)['pooler_output']

    def grid_features(self, x):
        if self.package == 'timm':
            return self.vision.forward_features(x)

        if self.package == 'openclip':
            features = {}
            def hook(module, input, output):
                features['output'] = output

            self.vision.trunk.blocks[-1].register_forward_hook(hook)
            self.vision(x)
            return features['output']

        elif self.package == 'transformers':
            return self.vision(x)['last_hidden_state']


if __name__ == '__main__':
    siglip_512 = {'input_size': 512,
              'output_dim': 768,
              'model_name': 'vit_base_patch16_siglip_512',
              'package': 'timm'}

    siglip2_256 = {'input_size': 256,
                   'output_dim': 768,
                   'model_name': 'hf-hub:timm/ViT-B-16-SigLIP2-256',
                   'package': 'openclip'}

    dinov2 = {'input_size': 224,
              'output_dim': 768,
              'model_name': 'facebook/dinov2-with-registers-base',
              'package': 'transformers'}

    cfg = json.load(open('D:\\modelos_v2\\encoder\\class3_siglip256\\experiment.json'))
    vision = Encoder(cfg)
    