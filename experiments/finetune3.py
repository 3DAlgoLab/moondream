import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moondream import detect_device, LATEST_REVISION
from accelerate import Accelerator
from IPython.display import display
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler


# setup params.
EPOCHS = 2
BATCH_SIZE = 4  # original value was 8
GRAD_ACCUM_STEPS = 1  # original value was 1
LR = 3e-5  # original value was 3e-5
USE_WANDB = False
VAL_STEPS = 100
ANSWER_EOS = "<END>"
IMG_TOKENS = 729


# global variables
accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS)
device = accelerator.device
dtype = torch.float16

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=device, dtype=dtype)

# if accelerator.is_main_process:
#     breakpoint()


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


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, max_steps, last_epoch=-1):
        self.max_steps = max_steps
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        x = self.last_epoch / self.max_steps
        if x < 0.1:
            return [0.1 * LR + 0.9 * LR * x / 0.1] * len(self.base_lrs)
        else:
            return [
                0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2
            ] * len(self.base_lrs)


def check_dataset(datasets):
    moondream.eval()
    sample = datasets["train"][0]
    display(sample["image"])

    for qa in sample["qa"]:
        print("Question:", qa["question"])
        print("Ground Truth:", qa["answer"])
        print(
            "Moondream:",
            moondream.answer_question(
                moondream.encode_image(sample["image"]),
                qa["question"],
                tokenizer=tokenizer,
            ),
        )


def collate_fn(batch):
    images = [sample["image"] for sample in batch]
    images = torch.stack(moondream.vision_encoder.preprocess(images))
    images = rearrange(images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        for qa in sample["qa"]:
            q_t = tokenizer(
                f"\n\nQuestion: {qa['question']}\n\nAnswer:", add_special_tokens=False
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
        images.to(dtype=dtype),
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )


def compute_loss(batch):
    images, tokens, labels, attn_mask = batch

    images = images.to(device)
    tokens = tokens.to(device)
    labels = labels.to(device)
    attn_mask = attn_mask.to(device)

    with torch.no_grad():
        img_embs = moondream.vision_encoder.encoder(images)
        img_embs = moondream.vision_encoder.projection(img_embs)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat(
        (tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1
    )

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss


def main():
    global USE_WANDB, moondream
    USE_WANDB = USE_WANDB and accelerator.is_main_process

    datasets = {
        "train": CaptchaDataset("train"),
        "val": CaptchaDataset("validation"),
        "test": CaptchaDataset("test"),
    }

    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        datasets["val"],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    )

    # prepare
    moondream.text_model.train()
    # moondream.text_model.transformer.gradient_checkpointing_enable()

    total_steps = EPOCHS * len(train_dataloader) // GRAD_ACCUM_STEPS
    optimizer = Adam8bit(
        [
            {"params": moondream.text_model.parameters()},
        ],
        lr=LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    scheduler = CustomScheduler(optimizer=optimizer, max_steps=total_steps)

    # apply accelerator
    moondream, optimizer, train_dataloader, val_dataloader, scheduler = (
        accelerator.prepare(
            moondream, optimizer, train_dataloader, val_dataloader, scheduler
        )
    )

    # moondream.vision_eoncoder = accelerator.prepare(moondream.vision_encoder)

    if USE_WANDB:
        import wandb

        wandb.init(
            project="moondream-ft",
            config={
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
                "LR": LR,
            },
        )

    i = 0
    val_loss_best = float("inf")
    for epoch in range(EPOCHS):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            moondream.train()
            i += 1

            loss = compute_loss(batch)
            # loss.backward()
            accelerator.backward(loss)

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            if i % VAL_STEPS == 0:
                # Calculate validation loss
                moondream.eval()
                val_loss = 0
                for val_batch in tqdm(val_dataloader, desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(val_batch).item()
                val_loss /= len(val_dataloader)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    moondream.save_pretrained("checkpoints/moondream-ft")

            if USE_WANDB:
                wandb.log(
                    {"loss/train": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                    | ({"loss/val": val_loss} if i % VAL_STEPS == 0 else {})
                )

    if USE_WANDB:
        wandb.finish()

    # Evaluation
    # Now that training has completed, let's inspect a few samples and calculate accuracy.
    if not accelerator.is_main_process:
        exit()

    moondream.from_pretrained("checkpoints/moondream-ft")
    moondream.eval()
    correct = 0

    for i, sample in enumerate(datasets["test"]):  # type: ignore
        md_answer = moondream.answer_question(
            moondream.encode_image(sample["image"]),
            sample["qa"][0]["question"],
            tokenizer=tokenizer,
        )

        if md_answer == sample["qa"][0]["answer"]:
            correct += 1

        if i < 3:
            display(sample["image"])
            print("Question:", sample["qa"][0]["question"])
            print("Ground Truth:", sample["qa"][0]["answer"])
            print("Moondream:", md_answer)

    print(f"\n\nAccuracy: {correct / len(datasets['test']) * 100:.2f}%")


if __name__ == "__main__":
    main()
