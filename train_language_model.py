from torch.utils.data import Dataset
import torch
from Bio import SeqIO
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from tokenizers.processors import BertProcessing
import sentencepiece as spm
import argparse
import yaml


### Arg parse
#   -infile train
#   -indir bpe model
#   -config
#   -outdir
parser = argparse.ArgumentParser(description='Train DNA language model from FASTA file.')
parser.add_argument('-input', metavar='INFILE', type=str, help='Path of DNA records for training language model.',required=True)
parser.add_argument('-tokenizer',
                    metavar='TOKENIZER_DIRECTORY',
                    type=str,
                    help='Directory containing trained BPE encoder. Must contain vocab.json and merges.txt')
parser.add_argument('-config',
                    metavar='CONFIG',
                    type=str,
                    help='Config file for language model. Must be YAML, see docs for details/examples.',
                    required=True)
parser.add_argument('-outdir',
                    metavar='OUTPUT',
                    type=str,
                    help='Path to output directory where trained model will be saved.',
                    required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # TODO code to validate config.

    #######
    # Convert the model to BERT type
    tokenizer = ByteLevelBPETokenizer(
        args.tokenizer + 'vocab.json',
        args.tokenizer + 'merges.txt'
    )
    tokenizer._tokenizer

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=config['model_params']['max_positional_embeddings']-2)
    tokenizer.save_model(args.tokenizer +'/as_BERT')

    #####
    # Load model config
    lm_config = RobertaConfig(**config['model_params'])

    # Instantiate model
    model = RobertaForMaskedLM(config=lm_config)
    print("Inst. model with {} parameters".format(model.num_parameters())

    # load data set
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path = args.input,
        block_size = config['dataset_params']['dataset_block_size']
    )

    training_args = TrainingArguments(
        output_dir=args.outdir,
        **config['training_params']
    )

    %%time
    trainer.train()
    trainer.save_model(args.outdir)
