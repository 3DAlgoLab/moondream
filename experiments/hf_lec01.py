# %%
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-cased")
unmasker("This course will teach you all about [MASK] models.", top_k=2)
# %%
dir(unmasker)
# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# %%
outputs

# %%
model.config.id2label
# %%
