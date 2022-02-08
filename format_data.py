from Bio import SeqIO
import yaml
import argparse


#TODO allow list of files instead of one file
parser = argparse.ArgumentParser(description='Format FASTA file for use with encoder.')
parser.add_argument('-input', metavar='INFILE', type=str, help='Path of FASTA file for formatting.',required=True)
parser.add_argument('-config',
                    metavar='CONFIG',
                    type=str,
                    help='Config file for formatter. Must be .yml'
                    required=True)
parser.add_argument('-outfile',
                    metavar='OUTPUT',
                    type=str,
                    help='Path to output file where formatted data set will be saved.',
                    required=True)

char_sets = {'dna_strict': {'A', 'C', 'G', 'T'},
             'dna_iupac': {'A', 'C', 'G', 'T',
                           'R', 'Y', 'S', 'W',
                           'K', 'M', 'B', 'D',
                           'H', 'V', 'N'},
             'rna_strict': {'A', 'C', 'G', 'U'},
             'rna_iupac': {'A', 'C', 'G', 'U',
                           'R', 'Y', 'S', 'W',
                           'K', 'M', 'B', 'D',
                           'H', 'V', 'N'},
             'aa' : {'A', 'C', 'D', 'E', 'F',
                     'G', 'H', 'I', 'K', 'L',
                     'M', 'N', 'P', 'Q', 'R',
                     'S', 'T', 'V', 'W', 'Y'
                    }
            }

def enforce_dna(seq, char_set, replace=True):
    seq = seq.upper()
    if replace:
        seq = seq.replace('U', "T")
    if set(seq) == char_set:
        return seq
    return False

def enforce_rna(seq, char_set, replace=True):
    seq = seq.upper()
    if replace:
        seq = seq.replace('T', "U")
    if set(seq) == char_set:
        return seq
    return False

def enforce_aa(seq, char_set, dummy=True):
    seq = seq.upper()
    if set(seq) == char_set:
        return seq
    return False

fmap = {
        'dna_strict': enforce_dna,
        'dna_iupac': enforce_dna,
        'rna_strict': enforce_rna,
        'rna_iupac': enforce_rna,
        'aa' : enforce_aa
        }

if __name__ == '__main__':
    args = parser.parse_args()

    fa = SeqIO.parse(args.input, 'fasta')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    char_set = char_sets[config['token_set']]
    enforce_func = fmap[config['token_set']]

    records_used = 0
    records_failed = 0
    with open(args.outfile, 'w') as f:
        for record in fa:
            clean = enforce_func(record.seq, char_set, config['rna_treatment'])
            if clean:
                f.write('{}\n'.format(r.seq))
                records_used += 1
            else:
                records_failed += 1
    print("Finished formatting.")
    print("{0} records were formatted. {1} records could not be used under policies {3}, {4}".format(records_used, records_failed, config['token_set'], config['rna_treatment']))
