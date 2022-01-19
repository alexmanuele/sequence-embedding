import sentencepiece as spm
import pyfastx
from io import BytesIO

import argparse
"""
## TODO:
- input max wordsize
- input outfile name
- possibly, allow other args?
- possibly, convert to HuggingFace automatically?
"""
def main():
    # Parse arguments: Requires n_words and outfile destination
    parser = argparse.ArgumentParser(description='Produce Byte Pair Encoder trained from .FASTA file.')

    parser.add_argument('-i',
                        metavar='INFILE',
                        type=str,
                        help='Path of FASTA file for training model.')


    args = parser.parse_args()

    assert args.i
    OUTFILE = "bpe_models/bpe_{}mer_dna_wordsize_max256.model"

    # Load all seqs into memory. This may be an issue depending on the machine.
    fa = pyfastx.Fasta(args.i)
    print("Fasta ", len(fa))
    #Make everything upper case DNA
    #seqs = ['{}'.format(record.seq).upper().replace('U', 'T') for record in fa]
    seqs = ['{}'.format(record.seq).upper() for record in fa]

    k_compares = [6,8]
    #Calculate target vocabulary sizes, in descending order
    k_compares = sorted(k_compares, reverse=True)
    vocab_sizes = [4**k for k in k_compares]


    for i, vocab_size in enumerate(vocab_sizes):
        #Create iterable for model input
        s_iter = iter(seqs)
        #Create bytes stream for model output
        model = BytesIO()
        #Train encoder.
        spm.SentencePieceTrainer.train(sentence_iterator=s_iter,
                                       model_writer=model,
                                       vocab_size=vocab_size,
                                       hard_vocab_limit="False",
                                       max_sentencepiece_length=256,
                                       character_coverage=1,
                                       model_type='bpe')
        #Save the model
        with open(OUTFILE.format(k_compares[i]), 'wb') as f:
            f.write(model.getvalue())

        #Encode the corpus to the reduced vocabulary
        sp = spm.SentencePieceProcessor(model_proto=model.getvalue())


    print("DONE.")

if __name__ == "__main__":
    main()
