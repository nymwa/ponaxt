from argparse import ArgumentParser
from .preproc import preproc
from .train import train

def parse_args():
    parser = ArgumentParser()
    first = parser.add_subparsers()

    preproc(first)
    train(first)

    return parser.parse_args()

