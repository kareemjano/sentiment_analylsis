from src.runners.train import trainer
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Sentiment Analysis Bert',
        usage='python train.py',
        description = "Train and evaluate a Bert model for sentiment analysis",
        add_help=True
    )

    parser.add_argument('-e', '--epochs', type=int, default=10, help='max number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='training rate') #8e-6
    parser.add_argument('-r', '--reg', type=float, default=1e-2, help='regularization') #1e-6
    parser.add_argument('-d', '--dropout', type=float, default=0.3, help="dropout rate")
    parser.add_argument('-tr', '--run_train', type=str, default="True", help='run training')
    parser.add_argument('-v', '--run_val', type=str, default="True", help='run validation')
    parser.add_argument('-te', '--run_test', type=str, default="True", help='run testing')
    parser.add_argument('-c', '--ckpt', type=str, default=None, help="load checkpoint to validate or test")
    parser.add_argument('-gpu', '--gpu', type=int, default=1, help="gpus to use. 0 will run on cpu.")
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = get_parser().parse_args()

    conf = {
        "batch_size": parser.batch_size,
        "model_name": "bert-base-cased",
        "max_input_len": 64,
        "data_src_file": "build/datasets/labelled_text.csv",
        "epochs": parser.epochs,
        "lr": parser.learning_rate,
        "weight_decay": parser.reg,
        "dropout_rate": parser.dropout,
    }
    do_run = [str2bool(v) for v in [parser.run_train,
                                    parser.run_val, parser.run_test]]

    trainer(conf=conf, run_train=do_run[0], run_test=do_run[2],
            run_val=do_run[1], ckpt=parser.ckpt, gpu=parser.gpu)