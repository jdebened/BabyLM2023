from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=256, truncate=True)
tokenizer.save_pretrained("./tokenizer/")
