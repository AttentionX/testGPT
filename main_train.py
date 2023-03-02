from datetime import datetime
from pathlib import Path
import torch
import wandb
from testgpt.gpt_v4 import GPTVer4
from testgpt.block_v4 import BlockVer4
from testgpt.multi_head_v2 import MultiHeadVer2
from tests.conftest import train, config, seed_everything


# push the model to overfit
config['max_iters'] = 30000
config['learning_rate'] = 0.001
config['embed_size'] = 256
config['n_layers'] = 3
config['dropout'] = 0.0001  # necessary for overfitting


def main():
    with wandb.init(entity="attentionx", project="testgpt", config=config) as run:
        seed_everything(1337)
        T, C, n_heads, dropout, V, n_layers = \
            config['block_size'], config['embed_size'], \
            config['n_heads'], config['dropout'], \
            config['vocab_size'], config['n_layers']
        # --- instantiate the model with the final version --- #
        contextualizer = torch.nn.Sequential(
            *[BlockVer4(MultiHeadVer2(T, C, n_heads), C, dropout) for _ in range(n_layers)])
        gpt = GPTVer4(contextualizer, V, T, C)
        # --- xavier initialization --- #
        for p in gpt.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        train(gpt, run)  # may take a while
        # --- persist to wandb --- #
        artifact = wandb.Artifact('gpt', type='model')
        save_dir = Path("out") / str(datetime.now())
        save_dir.mkdir(parents=True, exist_ok=True)
        gpt_bin_path = save_dir / 'gpt.bin'
        torch.save(gpt.state_dict(), gpt_bin_path)
        artifact.add_file(gpt_bin_path)
        run.log_artifact(artifact)


if __name__ == '__main__':
    main()
