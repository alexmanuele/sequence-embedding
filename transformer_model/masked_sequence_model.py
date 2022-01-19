from torch.utils.data import Dataset
import torch
from Bio import SeqIO
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import sentencepiece as spm
