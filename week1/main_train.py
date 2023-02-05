import json
import os
from datetime import datetime
from pathlib import Path
import torch
import argparse
from modeling_heads import HeadVer1, HeadVer2, HeadVer3, HeadVer4
from modeling_bigram import BigramLanguageModelVer1, BigramLanguageModelVer2


parser = argparse.ArgumentParser()
parser.add_argument("--bigram_ver", type=int, default=2,  choices=[1, 2])
parser.add_argument("--head_ver", type=int, default=4,  choices=[1, 2, 3, 4])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--block_size", type=int, default=8)
parser.add_argument("--max_iters", type=int, default=10000)
parser.add_argument("--eval_iters", type=int, default=200)
parser.add_argument("--eval_interval", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--n_embd", type=int, default=32)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--max_new_tokens", type=int, default=500)
args = parser.parse_args()
torch.manual_seed(args.seed)
os.environ['TOKENIZERS_PARALLELISM'] = "true"


# data loading
def get_batch(split: str, train_data, val_data):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([data[i:i+args.block_size] for i in ix])
    y = torch.stack([data[i+1:i+args.block_size+1] for i in ix])
    x, y = x.to(args.device), y.to(args.device)
    return x, y


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def main():
    if args.head_ver == 1:
        head = HeadVer1()
    elif args.head_ver == 2:
        head = HeadVer2()
    elif args.head_ver == 3:
        head = HeadVer3()
    elif args.head_ver == 4:
        head = HeadVer4(args.block_size, args.n_embd)
    else:
        raise ValueError("Invalid head version:" + args.head_ver)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    if args.bigram_ver == 1:
        model = BigramLanguageModelVer1(vocab_size)
    elif args.bigram_ver == 2:
        model = BigramLanguageModelVer2(vocab_size, head, args.n_embd)
    else:
        raise ValueError("Invalid bigram version:" + args.bigram_ver)

        # create a mapping from characters to integers
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [char2idx[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([idx2char[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    m = model.to(args.device)
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for i in range(args.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if i % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data)
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    completion = decode(m.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist())
    # --- persist to local --- #
    log_path = Path(__file__).resolve().parent / "logs" / str(datetime.now())
    log_path.mkdir(exist_ok=True)
    with open(log_path / "completion.txt", 'w') as fh:
        fh.write(completion)
    with open(log_path / "args.json", 'w') as fh:
        fh.write(json.dumps(vars(args)))


if __name__ == '__main__':
    main()
