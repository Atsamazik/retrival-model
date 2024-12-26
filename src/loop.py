from pathlib import Path

import typer
import torch
import transformers
from datasets import load_from_disk
from datasets import Dataset
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm import tqdm

from configurations.validation import load_config
from src.dataset import RetrievalDataset, RetrievalCollator, create_dataloader
from src.model import RetrievalModel
from src.trainable import RetrievalTrainable

app = typer.Typer()
config = load_config()
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVE_MODEL_PATH = PROJ_ROOT / "models"


def train(train_dataset: Dataset, eval_dataset: Dataset):

    collator = RetrievalCollator()

    train_dataset, eval_dataset = RetrievalDataset(train_dataset), RetrievalDataset(eval_dataset)
    train_dataloader = create_dataloader(train_dataset, collator)
    eval_dataloader = create_dataloader(eval_dataset, collator)

    model = RetrievalModel()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_dataloader) * 0.1),
        num_training_steps=len(train_dataloader)
    )

    trainable = RetrievalTrainable(model)

    model.freeze_base_model()

    for i, batch in enumerate(tqdm(train_dataloader, desc='Train Loop')):
        loss = trainable.compute_loss(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i == 300:
            model.unfreeze_base_model()

        if (i % 100 == 0) and i != 0:
            with torch.no_grad():
                model.eval()
                losses = []
                for eval_batch in tqdm(eval_dataloader, desc='Evaluation'):
                    loss_eval = trainable.compute_loss(eval_batch)
                    losses.append(loss_eval.item())

                save_file(model.state_dict(), SAVE_MODEL_PATH / f"retrieval-{i}.safetensors")
                model.train()


@app.command()
def main():
    data = load_from_disk(PROCESSED_DATA_DIR)
    data = data.train_test_split(shuffle=True, test_size=0.02, seed=config.seed)
    result_model = train(data['train'], data['test'])


if __name__ == '__main__':
    main()
