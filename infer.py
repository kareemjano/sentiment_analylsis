from src.runners.infer import infer
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Sentiment Analysis Bert',
        usage='python infer.py',
        description = "Infer the sentiment given a model checkpoint and text.",
        add_help=True
    )

    parser.add_argument('-t', '--text', type=str, help='text to infer sentiment of')
    parser.add_argument('-c', '--ckpt', type=str, help='ckpt to use')

    return parser

if __name__ == '__main__':
    parser = get_parser().parse_args()
    res = infer(parser.text, parser.ckpt)
    print(res)
