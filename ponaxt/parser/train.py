from ponaxt.train.main import train_main
from .model import parse_model_args

def parse_args(parser):
    parser.add_argument('--max-tokens', type = int, default = 8000)

    parser.add_argument('--label-smoothing', type = float, default = 0.01)
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--weight-decay', type = float, default = 0.01)
    parser.add_argument('--max-grad-norm', type = float, default = 1.0)
    parser.add_argument('--scheduler', default = 'linexp')
    parser.add_argument('--warmup-steps', type = int, default = 4000)
    parser.add_argument('--start-factor', type = float, default = 1.0)

    parser.add_argument('--mask-th', type = float, default = 0.15)
    parser.add_argument('--replace-th', type = float, default = 0.03)

    parser.add_argument('--epochs', type = int, default = 500)
    parser.add_argument('--step-interval', type = int, default = 4)
    parser.add_argument('--save-interval', type = int, default = 10)


def train(first):
    parser = first.add_parser('train')
    parse_args(parser)
    parse_model_args(parser)
    parser.set_defaults(handler = train_main)

