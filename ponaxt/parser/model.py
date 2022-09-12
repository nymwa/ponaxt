def parse_model_args(parser):
    parser.add_argument('--vocab', default = 'vocab.txt')
    parser.add_argument('--model')

    parser.add_argument('--hidden-dim', type = int, default = 128)
    parser.add_argument('--rnn-dim', type = int, default = 128)
    parser.add_argument('--num-layers', type = int, default = 32)

    parser.add_argument('--dropout', type = float, default = 0.3)
    parser.add_argument('--embed-dropout', type = float, default = 0.1)
    parser.add_argument('--no-share-embedding', action = 'store_true')
    parser.add_argument('--no-cuda', action = 'store_true')

