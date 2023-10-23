import torch
from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from pathlib import Path

# values from the tutorial
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

max_length = 126

# values I am using instead for memory limitations
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=max_length+2,
    num_attention_heads=2,
    num_hidden_layers=2,
    type_vocab_size=1,
)

modelloc = "./pre-model-7-8"

tokenizer = RobertaTokenizerFast.from_pretrained(modelloc, max_len=max_length, truncate=True)


model = RobertaForMaskedLM(config=config)

print("parameters: ", model.num_parameters())
# => 84 million parameters



class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = RobertaTokenizerFast.from_pretrained(modelloc, max_length=max_length)

        self.examples = []

        folder = "./preprocessed-data/"
        #folder = "./data/babylm_data/babylm_10M/"
        src_files = Path(folder).glob("*eval") if evaluate else Path(folder).glob("*train")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += tokenizer(lines,truncation=True,max_length=max_length)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


dataset = EsperantoDataset()

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=modelloc,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)



trainer.train()

trainer.save_model(modelloc)


