# %%
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

# %%
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
# %%
raw_train_dataset.features
# %%
raw_datasets["validation"][87]
# %%
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# %%
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs


# %%
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
