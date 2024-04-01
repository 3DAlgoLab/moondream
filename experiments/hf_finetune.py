# %%
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_dataset = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# %%
def tokenize_func(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_dataset.map(tokenize_func, batched=True)
data_collector = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
from transformers import TrainingArguments, AutoModelForSequenceClassification
from transformers import Trainer

training_args = TrainingArguments("test-trainer")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collector,
    tokenizer=tokenizer,
)


# %%
trainer.train()
# %%
import numpy as np

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# %%
preds = np.argmax(predictions.predictions, axis=-1)
# %%
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)


# %%
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# %%
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collector,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
# %%
