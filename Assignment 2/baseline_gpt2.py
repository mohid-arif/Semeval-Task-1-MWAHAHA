import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
import pandas as pd
from data_utils import HumorDataset, build_prompt
import argparse
import os
from tqdm import tqdm

def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # Load CSV and build prompts
    # -----------------------
    df = pd.read_csv(args.data, sep="\t")

    if "prompt" not in df.columns:
        print("Building prompts dynamically...")
        df["prompt"] = df.apply(build_prompt, axis=1)
        df.to_csv(args.data, sep="\t", index=False)

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset & Loader
    dataset = HumorDataset(args.data, tokenizer=tokenizer, max_length=args.max_length, model_type="gpt2")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.resize_token_embeddings(len(tokenizer))

    opt = AdamW(model.parameters(), lr=5e-5)

    # -----------------------
    # Training
    # -----------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            opt.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} loss: {total_loss/len(loader):.4f}")

    # -----------------------
    # Generation
    # -----------------------
    model.eval()
    preds = []

    for prompt in df["prompt"].tolist():
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        out = model.generate(
            **enc,
            max_length=args.max_length,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # remove prompt prefix
        text = text.replace(prompt, "").strip()
        preds.append(text)

    # Save predictions
    os.makedirs(args.out_dir, exist_ok=True)
    df["prediction"] = preds
    df.to_csv(os.path.join(args.out_dir, "gpt2_preds.csv"), index=False)

    # Save model
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"Saved GPT-2 model and tokenizer to {args.save_model}")

    print("Done GPT-2 baseline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=1)  # tiny for 495MB VRAM
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_model", type=str, default=None)
    args = parser.parse_args()
    main(args)
