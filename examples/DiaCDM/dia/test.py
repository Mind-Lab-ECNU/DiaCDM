from transition_amr_parser.parse import AMRParser

# Download and save a model named AMR3.0 to cache
parser = AMRParser.from_pretrained('AMR3-structbart-L', dict_dir="/home/ai4learning/.cache/torch/pytorch_fairseq/")
tokens, positions = parser.tokenize('The girl travels and visits places')

# Use parse_sentence() for single sentences or parse_sentences() for a batch
annotations, machines = parser.parse_sentence(tokens)

# Print Penman notation
print(annotations)


