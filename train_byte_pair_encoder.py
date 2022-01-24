from byte_pair_encoding import bpe
import argparse
import yaml

### Arg parse
#   -infile
#   -config
#   -outdir
parser = argparse.ArgumentParser(description='Train Byte Pair Encoder from FASTA file.')
parser.add_argument('-input', metavar='INFILE', type=str, help='Path of FASTA file for traing BPE model.',required=True)
parser.add_argument('-config',
                    metavar='CONFIG',
                    type=str,
                    help='Config file for BPE model. Must be YAML, see docs for details/examples.',
                    required=True)
parser.add_argument('-outdir',
                    metavar='OUTPUT',
                    type=str,
                    help='Path to output directory where trained model will be saved.',
                    required=True)

if __name__ == '__main__':
    parser.parse_args()
    ############################################################
    # TODO: Consider adding parsing for valid configfile here. #
    #
    # def parse_config(config):
    #    pass
    ############################################################
    bpe.train_model(args.input, args.config, args.outdir)
    
