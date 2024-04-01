# %%
from transformers import BertConfig, BertModel


config = BertConfig()
model = BertModel(config)
print(config)
# %%
model = BertModel.from_pretrained("bert-base-cased")
model.save_pretrained("test_check")  # type: ignore
# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple."
tokens = tokenizer.tokenize(sequence)
print(tokens)
# %%
ids = tokenizer.convert_tokens_to_ids(tokens)
ids
# %%
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
decoded_string
# %%
