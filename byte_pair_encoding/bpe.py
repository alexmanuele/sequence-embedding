import sentencepiece as spm
import pyfastx
from io import BytesIO
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os

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
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outstring = "{0}/{1}_model_{2}vocab_{3}maxlength.model".format(outdir,
                                                            config['model_type'],
                                                            config['vocab_size'],
                                                            config['max_sentencepiece_length'],)
    with open(outstring, 'wb') as f:
        f.write(model.getvalue())

    #convert the model to HuggingFace type
    ###### adapted from https://github.com/huggingface/tokenizers/blob/master/bindings/python/scripts/sentencepiece_extractor.py
    extractor = SentencePieceExtractor(outstring)
    hf_model_dest = "{}/as_hugging_face".format(outdir)
    vocab_file = hf_model_dest+'vocab.json'
    merges_file = hf_model_dest + 'merges.txt'
    if not os.path.exists(hf_model_dest):
        os.makedirs(hf_model_dest)

    with open(vocab_file, 'w') as vocab_f:
        with open(merges_file, 'w') as merges_f:
            vocab, merges = extractor.extract
            dump(vocab, vocab_f)
            merges_f.writelines(map(lambda x: f"{x[0]} {x[1]}{os.linesep}", merges))

#From HuggingFace:
# https://github.com/huggingface/tokenizers/blob/master/bindings/python/scripts/sentencepiece_extractor.py
class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models.
    https://github.com/google/sentencepiece
    """

    def __init__(self, model: str):
        # Get SentencePiece
        self.sp = SentencePieceProcessor()
        self.sp.Load(model)

    def extract(self) -> Tuple[Dict[str, int], List[Tuple]]:
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in trange(sp.GetPieceSize())}

        # Merges
        merges = []
        for piece_l in tqdm(vocab.keys(), total=sp.GetPieceSize()):
            for piece_r in vocab.keys():
                merge = f"{piece_l}{piece_r}"
                piece_id = vocab.get(merge, None)
                if piece_id:
                    merges += [(piece_l, piece_r, piece_id)]
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(val[0], val[1]) for val in merges]

        return vocab, merges
