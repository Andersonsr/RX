import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from peft import LoraConfig, get_peft_model


class Decoder(torch.nn.Module):
    def __init__(self, model_name, vision_dim, mapper):
        super(Decoder, self).__init__()

        if 'opt' in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.ignore_id = -100

        elif 'llama' in model_name:
            assert 'HF_TOKEN' in os.environ.keys(), 'HF_TOKEN environment variable not set'
            # login(token=os.environ['HF_TOKEN'])
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

            # <|eot_id|> = 128009, <|end_of_text|> = 128001
            self.tokenizer.pad_token_id = 128009
            self.model.generation_config.pad_token_id = 128009

            self.tokenizer.eos_token_id = 128001
            self.model.generation_config.eos_token_id = 128001
            self.ignore_id = -100

        else:
            raise ValueError('{} not supported'.format(model_name))

        self.dim = self._get_hidden_size()
        self.mapper = mapper(vision_dim, self.dim)

        logging.debug(f'hidden size: {self.dim}')
        logging.debug(f'BOS token id: {self.tokenizer.bos_token_id}')
        logging.debug(f'EOS token: {self.tokenizer.eos_token}')
        logging.debug(f'EOS token id: {self.tokenizer.eos_token_id}')
        logging.debug(f'PAD token: {self.tokenizer.pad_token}')
        logging.debug(f'PAD token id: {self.tokenizer.pad_token_id}')

    def forward(self, vis_tokens, texts):
        texts = [text+self.tokenizer.eos_token for text in texts]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # ids to input embeddings
        input_embeddings = self.get_input_embeds(input_ids)
        vision_embeddings = self.mapper(vis_tokens)
        input_embeddings = torch.concat([vision_embeddings, input_embeddings], dim=1)

        # append vision attention mask
        vis_mask = torch.ones(vis_tokens.shape[:2], dtype=torch.long)
        attention_mask = torch.concat([attention_mask, vis_mask], dim=1)

        # append vision labels
        vis_labels = torch.ones(vis_tokens.shape[:2], dtype=torch.long) * self.ignore_id
        input_ids = torch.concat([vis_labels, input_ids], dim=1)

        logging.debug('input embeddings {}'.format(input_embeddings.shape))
        logging.debug('ids shape {}'.format(input_ids.shape))
        logging.debug('attention mask shape {}'.format(attention_mask.shape))

        return self.model(inputs_embeds=input_embeddings, attention_mask=attention_mask, labels=input_ids)

    def get_input_embeds(self, input_ids):
        with torch.no_grad():
            embeddings_layer = self.model.get_input_embeddings()
            return embeddings_layer(input_ids)

    def _get_hidden_size(self):
        ids = self.tokenizer("prompt", return_tensors="pt").input_ids.squeeze(0)
        embeddings = self.model.get_input_embeddings()
        return embeddings(ids).shape[1]

    def lora_model(self, r, alpha, dropout):
        for param in self.model.parameters():
            param.requires_grad = False

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",

        )
        self.model = get_peft_model(self.model, config).to(self.fp)

    def generate(self):
        raise NotImplementedError


if __name__ == '__main__':
    decoder = Decoder('facebook/opt-350m', 768).cpu()
    decoder.forward(torch.rand(2, 256, 768), ['um texto qualquer', 'mais um outro texto qualquer'])
