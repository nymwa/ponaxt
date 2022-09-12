from ponaxt.preproc.main import preproc_main

def preproc(first):

    def command(args):
        preproc_main(args.train, args.valid, args.max_len)

    parser = first.add_parser('preproc')
    parser.add_argument('--train', default = 'train.txt')
    parser.add_argument('--valid', default = 'valid.txt')
    parser.add_argument('--max-len', type = int, default = 120)
    parser.set_defaults(handler = command)

