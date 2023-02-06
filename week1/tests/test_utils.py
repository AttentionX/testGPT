import os
from pathlib import Path
from typing import Union
import torch
import yaml
from ..src import GPTVer1, GPTVer2

# --- load config --- #
with open(Path(__file__).resolve().parent / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# -- load input.txt --- #
with open(Path(__file__).resolve().parent / "input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# --- prep train & test sets --- #
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
config['vocab_size'] = vocab_size
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char2idx[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([idx2char[i] for i in l])  # decoder: take a list of integers, output a string
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# --- just to make the huggingface api happy --- #
os.environ['TOKENIZERS_PARALLELISM'] = "true"


def get_batch(split: str):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([data[i:i + config['block_size']] for i in ix])
    y = torch.stack([data[i + 1:i + config['block_size'] + 1] for i in ix])
    x, y = x.to(config['device']), y.to(config['device'])
    return x, y


@torch.no_grad()
def estimate_loss(model: torch.nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(lm: Union[GPTVer1, GPTVer2]):
    lm = lm.to(config['device'])
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(lm.parameters(), lr=config['learning_rate'])
    for i in range(config['max_iters']):
        # every once in a while evaluate the loss on train and val sets
        if i % config['eval_interval'] == 0:
            losses = estimate_loss(lm)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = lm(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return lm


def generate(lm: Union[GPTVer1, GPTVer2], context: str, max_new_tokens: int) -> str:
    # generate text
    lm.eval()
    context = torch.tensor(encode(context), dtype=torch.long).to(config['device'])
    context = context.unsqueeze(0)  # add batch dimension
    completion = lm.generate(context, max_new_tokens)
    completion = decode(completion.squeeze().tolist())
    return completion
