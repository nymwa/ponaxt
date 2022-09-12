from .util import (
        make_tokens,
        get_train_sents,
        get_valid_sents,
        sents_to_data)
from ponaxt.vocab import Vocab
from seriejo import SeriejoWriter
from pathlib import Path
from logging import getLogger
logger = getLogger(__name__)

def write_vocab(tokens):
    with open('vocab.txt', 'w') as f:
        for x in tokens:
            print(x, file = f)


def make_raw(base, name, sents):
    Path(base).mkdir(parents = True, exist_ok = True)

    with open('{}/{}.txt'.format(base, name), 'w') as f:
        for x in sents:
            print(x, file = f)


def make_seriejo(base, name, data):
    Path(base).mkdir(parents = True, exist_ok = True)

    with SeriejoWriter('{}/{}'.format(base, name)) as f:
        for x in data:
            f.write(x)

    logger.info('Write Seriejo ({}/{}): {}'.format(base, name, len(data)))


def preproc_main(train, valid, max_len):
    train_sents = get_train_sents(train, max_len)
    tokens = make_tokens(train_sents)
    write_vocab(tokens)

    vocab = Vocab(tokens)
    valid_sents = get_valid_sents(valid)

    make_raw('data', 'train', train_sents)
    make_raw('data', 'valid', valid_sents)

    train_data = sents_to_data(vocab, train_sents)
    valid_data = sents_to_data(vocab, valid_sents)

    make_seriejo('data', 'train', train_data)
    make_seriejo('data', 'valid', valid_data)

