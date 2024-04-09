from torch.utils.data import Dataset
import argparse
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator, DistributedType
from einops import rearrange
import time
from bitsandbytes.optim import Adam8bit

from torchvision.transforms.v2 import (
    Compose,
    Resize,
    InterpolationMode,
    ToImage,
    ToDtype,
    Normalize,
)

MAX_GPU_BATCH_SIZE = 8
MD_REVISION = "2024-04-02"
ANSWER_EOS = "<|endoftext|>"
# Number of tokens used to represent each image.
IMG_TOKENS = 729

vision_preprocess = Compose(
    [
        Resize(size=(378, 378), interpolation=InterpolationMode.BICUBIC),
        ToImage(),
        ToDtype(torch.float16, scale=True),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class CaptchaDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset(
            "project-sloth/captcha-images",
            revision="refs/convert/parquet",  # type: ignore
        )[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"],  # Should be a PIL image
            "qa": [
                {
                    "question": "What does the text say?",
                    "answer": sample["solution"],
                }
            ],
        }


def get_dataloaders(accelerator: Accelerator, batch_size=16):

    def collate_fn(batch):
        images = [sample["image"] for sample in batch]
        images = torch.stack(vision_preprocess(images))
        images = rearrange(
            images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14
        )

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample["qa"]:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False,
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images,
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision=MD_REVISION
        )

        datasets = {
            "train": CaptchaDataset("train"),
            "val": CaptchaDataset("validation"),
            # "test": CaptchaDataset("test"),
        }

    train_dataloader = DataLoader(
        datasets["train"],  # type: ignore
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        datasets["val"],  # type: ignore
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader


def compute_loss(model, batch, accelerator: Accelerator):
    images, tokens, labels, attn_mask = batch
    model = accelerator.unwrap_model(model)
    with torch.no_grad():
        img_embs = model.vision_encoder.encoder(images)
        img_embs = model.vision_encoder.projection(img_embs)

    tok_embs = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1
    )

    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)

    # Sample hyper-parameters for learning rate,
    # batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    # metric = evaluate.load("glue", "mrpc")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if (
        batch_size > MAX_GPU_BATCH_SIZE
        and accelerator.distributed_type != DistributedType.XLA
    ):
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision=MD_REVISION,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        # device_map={"": DEVICE},
        # device_map="auto",
    )

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=(len(train_dataloader) * num_epochs)
        // gradient_accumulation_steps,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    assert model.vision_encoder

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # assert model.vision_encoder

    # assert model.text_model
    # model.text_model.transformer.gradient_checkpointing_enable()

    # Now we train the model
    for epoch in range(num_epochs):
        # model.text_model.train()
        model.train()

        for step, batch in enumerate(train_dataloader):
            loss = compute_loss(model, batch, accelerator)
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        val_loss = 0
        for step, batch in enumerate(eval_dataloader):

            with torch.no_grad():
                val_loss += compute_loss(model, batch, accelerator).item()

            val_loss /= len(eval_dataloader)

        accelerator.print(
            f"Epoch {epoch + 1}/{num_epochs} - Validation loss: {val_loss}"
        )


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )

    seed = time.time_ns() % 1000
    config = {
        "lr": 3e-5,
        "num_epochs": 10,
        "seed": seed,
        "batch_size": 2,
    }

    args = parser.parse_args()
    training_function(config, args)


if __name__ == "__main__":
    main()
