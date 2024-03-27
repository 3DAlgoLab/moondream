from datasets import load_dataset

d = load_dataset(
    "project-sloth/captcha-images",
    revision="refs/convert/parquet",
)

dt = d["train"]
print(dt[0])
