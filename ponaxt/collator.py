import torch
from torch.nn.utils.rnn import pad_sequence as pad
from ponaxt.batch import Batch

class Collator:

    def __init__(self, vocab, mask_th = 0.15, replace_th = 0.03):
        self.vocab = vocab
        self.mask_th = mask_th
        self.replace_th = replace_th

    def __call__(self, batch):
        batch = [[self.vocab.bos] + sent + [self.vocab.eos] for sent in batch]
        inputs = [torch.tensor(sent) for sent in batch]
        outputs = [torch.tensor(sent) for sent in batch]
        lengths = [len(sent) for sent in batch]

        inputs = pad(inputs, padding_value = self.vocab.pad)
        outputs = pad(outputs, padding_value = -100)

        rand_tensor = torch.rand(inputs.shape)
        rand_token = torch.randint(self.vocab.msk, len(self.vocab), inputs.shape)
        normal_token = inputs > self.vocab.msk
        position_to_mask = (rand_tensor < self.mask_th) & normal_token
        position_to_replace = (rand_tensor < self.replace_th) & normal_token

        inputs.masked_fill_(position_to_mask, self.vocab.msk)
        inputs.masked_scatter_(position_to_replace, rand_token)
        outputs.masked_fill_(~position_to_mask, -100)

        return Batch(inputs, outputs, lengths)

