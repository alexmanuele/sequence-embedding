import sentencepiece as spm
import pyfastx
from io import BytesIO

def train_model(infile, config, outdir):
    #load the file. Expected to be a fasta.
    fa = pyfastx.Fasta(infile)
    seqs = ['{}'.foramt(record.seq).upper() for record in fa]
    #make an iterable that is compatible with sentencepiece
    s_iter(seqs)
    #Create bytes stream for model output
    model= BytesIO()
    # train the encoder
    #Config is a dict of params. Will be checked for validity from runner program
    spm.SentencePieceTrainer.train(sentence_iterator=s_iter, **config)
    # Save the trained model
    outstring = "{0}/{1}_model_{2}vocab_{3}maxlength.model".format(outdir,
                                                            config['model_type'],
                                                            config['vocab_size'],
                                                            config['max_sentencepiece_length'],)
    with open(outstring, 'wb') as f:
        f.write(model.getvalue())
