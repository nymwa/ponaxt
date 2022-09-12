from .preproc import LMPreproc
from collections import Counter
from logging import getLogger
logger = getLogger(__name__)

def sents_to_data(vocab, sents):

    def sent_to_data(sent):
        sent = sent.split()
        sent = [vocab(token) for token in sent]
        return sent

    return [sent_to_data(sent) for sent in sents]


def get_train_sents(train_path, max_len):
    preproc = LMPreproc()

    with open(train_path) as f:
        sents = [preproc(sent) for sent in f]
    logger.info('loaded train: {}'.format(len(sents)))

    sents = [
        sent
        for sent
        in sents
        if 1 <= len(sent.split()) <= max_len]
    logger.info('filtered train: {}'.format(len(sents)))
    return sents


def get_valid_sents(valid_path):
    preproc = LMPreproc()

    with open(valid_path) as f:
        sents = [preproc(sent) for sent in f]
    logger.info('loaded valid: {}'.format(len(sents)))
    return sents


def make_tokens(train_sents):
    freq = Counter([
        word
        for sent
        in train_sents
        for word
        in sent.split()
        ]).most_common()
    tokens = [w for w, f in freq if w != '<unk>']
    tokens = ['<pad>', '<bos>', '<eos>', '<msk>', '<unk>'] + tokens
    logger.info('Make Tokens -> vocab size: {}'.format(len(tokens)))
    return tokens

