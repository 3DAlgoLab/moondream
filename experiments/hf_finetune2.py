# %%
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_dataset = load_dataset("glue", "sst2")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# %%
raw_dataset


# %%
def tokenize_func(example):
    return tokenizer(example["sentence"], truncation=True)


tokenized_datasets = raw_dataset.map(tokenize_func, batched=True)
data_collector = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer


# %%
training_args = TrainingArguments("checkout/test-trainer2")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collector,
    tokenizer=tokenizer,
)

trainer.train()
# %%
import numpy as np
import evaluate

# %%
predictions = trainer.predict(tokenized_datasets["validation"])
# %%

preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "sst2")
metric.compute(predictions=preds, references=predictions.label_ids)

# %%
