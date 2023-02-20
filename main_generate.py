import torch
from learngpt.gpt_v4 import GPTVer4
from learngpt.block_v4 import BlockVer4
from learngpt.multi_head_v2 import MultiHeadVer2
from tests.conftest import generate, train, config

torch.manual_seed(1337)


def main():
    T, C, n_heads, dropout = config['block_size'], config['embed_size'], config['n_heads'], config['dropout']
    contextualizer = torch.nn.Sequential(
        *[BlockVer4(MultiHeadVer2(T, C, n_heads), C, dropout) for _ in range(config['n_layers'])])
    gpt = GPTVer4(contextualizer, config['vocab_size'], T, C)
    config['max_iters'] = 15000
    train(gpt)  # may take a while
    completion = generate(gpt, "A", 1000)
    print(completion)


if __name__ == '__main__':
    main()
