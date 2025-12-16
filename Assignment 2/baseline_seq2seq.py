# baseline_seq2seq.py
"""
Minimal Bi-LSTM encoder -> LSTM decoder with Bahdanau attention.
Adjusted for small dataset (5k) and low VRAM (495MB).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
import pandas as pd
from data_utils import load_semeval_dataset, train_val_test_split, save_preds
import argparse
import os
from collections import Counter
import random
from tqdm import tqdm

# --- dataset & vocab helpers ---
class Vocab:
    def __init__(self, min_freq=1, max_size=10000):
        self.word2idx = {"<pad>":0, "<s>":1, "</s>":2, "<unk>":3}
        self.idx2word = {0:"<pad>",1:"<s>",2:"</s>",3:"<unk>"}
        self.min_freq = min_freq
        self.max_size = max_size
        self.counter = Counter()

    def add_sentence(self, sent):
        for t in tokenizer.tokenize(sent.lower()):
            self.counter[t] += 1

    def build(self):
        most = [w for w,c in self.counter.most_common(self.max_size) if c>=self.min_freq]
        for w in most:
            if w not in self.word2idx:
                i = len(self.word2idx)
                self.word2idx[w] = i
                self.idx2word[i] = w

    def encode(self, sent):
        return [self.word2idx.get(w, self.word2idx["<unk>"]) for w in tokenizer.tokenize(sent.lower())]

    def decode(self, idxs):
        return " ".join([self.idx2word.get(i, "<unk>") for i in idxs])

class JokeDataset(Dataset):
    def __init__(self, df, vocab, max_src=64, max_tgt=64):
        self.df = df
        self.vocab = vocab
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.vocab.encode(row['prompt'])[:self.max_src]
        tgt = self.vocab.encode(row['target'])[:self.max_tgt]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in tgts) + 2
    pad = 0
    s_batch = torch.full((len(batch), max_s), pad, dtype=torch.long)
    t_batch = torch.full((len(batch), max_t), pad, dtype=torch.long)
    for i,s in enumerate(srcs):
        s_batch[i,:len(s)] = s
    for i,t in enumerate(tgts):
        t_batch[i,0] = 1
        t_batch[i,1:1+len(t)] = t
        t_batch[i,1+len(t)] = 2
    return s_batch, t_batch

# --- model ---
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hid=128, n_layers=1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.hid = hid
        self.bid = bidirectional

    def forward(self, src, src_lens=None):
        emb = self.embedding(src)
        outputs, (h,c) = self.rnn(emb)
        return outputs, (h,c)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        B, T, H = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, enc_hid=256, dec_hid=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + enc_hid, dec_hid, batch_first=True)
        self.attn = BahdanauAttention(enc_hid, dec_hid)
        self.out = nn.Linear(dec_hid, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        emb = self.embedding(input_token).unsqueeze(1)
        h = hidden[0][-1]
        context, attn = self.attn(h, encoder_outputs)
        rnn_input = torch.cat((emb, context.unsqueeze(1)), dim=2)
        output, (h_n, c_n) = self.rnn(rnn_input, hidden)
        logits = self.out(output.squeeze(1))
        return logits, (h_n, c_n), attn

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, enc_hid=128, dec_hid=128):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb_dim, enc_hid//2, bidirectional=True)
        self.decoder = Decoder(vocab_size, emb_dim, enc_hid, dec_hid)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_len=64):
        batch_size = src.size(0)
        encoder_outputs, _ = self.encoder(src)
        device = src.device
        outputs = []
        input_token = torch.full((batch_size,), 1, dtype=torch.long, device=device)
        hidden = (torch.zeros(1, batch_size, 128, device=device),
                  torch.zeros(1, batch_size, 128, device=device))
        for t in range(max_len):
            logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs.append(logits.unsqueeze(1))
            top1 = logits.argmax(dim=1)
            if tgt is not None and random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = top1
        outputs = torch.cat(outputs, dim=1)
        return outputs

# --- training ---
def train_loop(model, dataloader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    for src, tgt in tqdm(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        opt.zero_grad()
        out = model(src, tgt, teacher_forcing_ratio=0.5, max_len=tgt.size(1)-1)
        loss = criterion(out.view(-1, out.size(-1)), tgt[:, :out.size(1)].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def generate_from_model(model, vocab, src_sentence, device, max_len=64):
    model.eval()
    with torch.no_grad():
        src_ids = torch.tensor([vocab.encode(src_sentence)], dtype=torch.long).to(device)
        outputs = model(src_ids, tgt=None, teacher_forcing_ratio=0.0, max_len=max_len)
        preds = outputs.argmax(dim=2).squeeze(0).tolist()
    words = []
    for p in preds:
        w = vocab.idx2word.get(p, "<unk>")
        if w == "</s>":
            break
        words.append(w)
    return " ".join(words)

def main(args):
    df = load_semeval_dataset(args.data)
    train, val, test = train_val_test_split(df)

    vocab = Vocab(min_freq=1, max_size=10000)
    for s in train['prompt'].tolist() + train['target'].tolist():
        vocab.add_sentence(s)
    vocab.build()

    train_ds = JokeDataset(train, vocab, max_src=64, max_tgt=64)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Seq2Seq(len(vocab.word2idx), emb_dim=128, enc_hid=128, dec_hid=128).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        loss = train_loop(model, train_loader, opt, criterion, device)
        print(f"Epoch {epoch} loss {loss:.4f}")

    preds, prompts, ids = [], [], []
    for i, row in test.reset_index().iterrows():
        g = generate_from_model(model, vocab, row['prompt'], device, max_len=args.max_len)
        preds.append(g)
        prompts.append(row['prompt'])
        ids.append(i)

    os.makedirs(args.out_dir, exist_ok=True)
    save_preds(os.path.join(args.out_dir, "seq2seq_preds.csv"), ids, prompts, preds, refs=None)
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved seq2seq model to {args.save_model}")
    print("Done seq2seq baseline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--save_model", type=str, default=None, help="Optional path to save trained model")
    args = parser.parse_args()
    main(args)
