#! pip install tokenizers

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

#folder = "./data/babylm_data/babylm_10M/"
folder = "./preprocessed-data/"
paths = [str(x) for x in Path(folder).glob("**/*.train")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("./pre-model-7-8")




from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./pre-model-7-8", max_len=126, truncate=True)
tokenizer.save_pretrained("./pre-model-7-8/")

