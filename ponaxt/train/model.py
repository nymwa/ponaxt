import torch
from pathlib import Path
from ponaxt.model.xt import PonaXT

from logging import getLogger
logger = getLogger(__name__)

def get_xt_model(vocab, args):
    model = PonaXT(
            len(vocab),
            args.hidden_dim,
            args.rnn_dim,
            args.dropout,
            args.embed_dropout,
            args.num_layers,
            padding_idx = vocab.pad)

    if not args.no_share_embedding:
        # なぜか式の左右を逆にすると動かない
        # パラメータの初期化の問題かもしれないが謎
        model.embedding.token_embedding.weight = model.fc.weight

    logger.info('#params : {} ({})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.model is not None:
        model_path = Path(args.model).resolve()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info('model <- {}'.format(model_path))

    if not args.no_cuda:
        model = model.cuda()

    return model

