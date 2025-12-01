# baseline_ngram.py
"""
Simple n-gram baseline.
Train: build trigram counts and use simple backoff Kneser-Ney-ish smoothing.
Generate: sample via next-word probabilities conditioned on last 2 tokens.
This is a minimal, easy-to-run classical baseline.
"""
import math, random
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
import pandas as pd
from data_utils import load_semeval_dataset, train_val_test_split, save_preds
import argparse
import os

class NgramModel:
    def __init__(self, n=3, unk_threshold=1):
        self.n = n
        self.counts = [defaultdict(int) for _ in range(n)]
        self.context_counts = [defaultdict(int) for _ in range(n)]
        self.vocab = set()
        self.total = 0
        self.unk_threshold = unk_threshold

    def train(self, sents):
        for s in sents:
            tokens = ["<s>"] + [t.lower() for t in tokenizer.tokenize(s)] + ["</s>"]
            for token in tokens:
                self.vocab.add(token)
            for k in range(1, self.n+1):
                for i in range(len(tokens)-k+1):
                    ngram = tuple(tokens[i:i+k])
                    self.counts[k-1][ngram] += 1
                    context = tuple(tokens[i:i+k-1]) if k>1 else ()
                    self.context_counts[k-1][context] += 1
            self.total += 1

    def score_next(self, context):
        """
        Return distribution over next words given context (list of last n-1 tokens)
        Use simple backoff: try trigram -> bigram -> unigram with add-k smoothing
        """
        max_k = min(self.n-1, len(context))
        for k in range(max_k, 0, -1):
            ctx = tuple(context[-k:])
            candidates = {}
            # consider all words observed with this context
            for ngram,count in self.counts[k].items():
                if ngram[:-1] == ctx:
                    candidates[ngram[-1]] = count
            if candidates:
                # normalize
                total = sum(candidates.values())
                for w in list(candidates.keys()):
                    candidates[w] /= total
                return candidates
        # unigram fallback
        candidates = {}
        for (w,),c in self.counts[0].items():
            candidates[w] = c
        total = sum(candidates.values())
        for w in candidates:
            candidates[w] /= total
        return candidates

    def generate(self, prompt, max_len=30):
        tokens = [t.lower() for t in tokenizer.tokenize(prompt)]
        out = tokens.copy()
        for _ in range(max_len):
            ctx = out[-(self.n-1):] if len(out)>0 else []
            dist = self.score_next(ctx)
            # sample
            words = list(dist.keys())
            probs = list(dist.values())
            next_word = random.choices(words, probs)[0]
            if next_word == "</s>":
                break
            out.append(next_word)
        return " ".join(out)

def main(args):
    df = load_semeval_dataset(args.data)
    train, val, test = train_val_test_split(df)
    sents = train['target'].tolist()
    model = NgramModel(n=3)
    model.train(sents)
    preds = []
    prompts = []
    ids = []
    for i,row in test.reset_index().iterrows():
        prompt = row['prompt']
        gen = model.generate(prompt, max_len=args.max_len)
        preds.append(gen)
        prompts.append(prompt)
        ids.append(i)
    os.makedirs(args.out_dir, exist_ok=True)
    save_preds(os.path.join(args.out_dir, "ngram_preds.csv"), ids, prompts, preds, refs=None)
    if args.save_model:
        import pickle
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        with open(args.save_model, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved ngram model to {args.save_model}")
    print("Done ngram baseline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to dataset csv/tsv")
    parser.add_argument("--out_dir", default="outputs", help="where to write preds")
    parser.add_argument("--max_len", type=int, default=20)
    # add to the parser in baseline_ngram.py
    parser.add_argument("--save_model", type=str, default=None, help="Path to save the trained model")
    
    args = parser.parse_args()

    main(args)
