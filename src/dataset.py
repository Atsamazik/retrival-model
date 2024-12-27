from pathlib import Path

import datasets
import torch.nn.functional as F
import torch.utils.data
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from configurations.validation import load_config

config = load_config()

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def pad_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(x) for x in tensors)
    return torch.stack([F.pad(x, pad=(0, max_len - len(x)), mode='constant', value=0) for x in tensors])


class RetrievalDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: datasets.Dataset):
        self._dataset = dataset
        self._tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)

    def __getitem__(self, index) -> dict[str, dict[str, torch.Tensor]]:
        item = self._dataset[index]

        question = "query: " + item['question']
        answers = item['answers']

        score = item['score']

        positive_answer = "query: Unfortunately, I do not know the answer to this question."
        negative_answer = None
        min_negative_score = 0

        for answer in answers:
            if answer['score'] == score and score >= 0:
                positive_answer = "passage: " + answer['body']
            if answer['score'] < min_negative_score:
                min_negative_score = answer['score']
                negative_answer = "passage: " + answer['body']

        if negative_answer is None:
            negative_answer = positive_answer

        anchor_enc = self._tokenizer.encode_plus(question,
                                                 max_length=512,
                                                 truncation="longest_first").encodings[0]
        positive_enc = self._tokenizer.encode_plus(positive_answer,
                                                   max_length=512,
                                                   truncation="longest_first").encodings[0]
        negative_enc = self._tokenizer.encode_plus(negative_answer,
                                                   max_length=512,
                                                   truncation="longest_first").encodings[0]
        return {
            'query': {
                'input_ids': torch.tensor(anchor_enc.ids, dtype=torch.long),
                'attention_mask': torch.tensor(anchor_enc.attention_mask, dtype=torch.long),
            },
            'positive': {
                'input_ids': torch.tensor(positive_enc.ids, dtype=torch.long),
                'attention_mask': torch.tensor(positive_enc.attention_mask, dtype=torch.long),
            },
            'negative': {
                'input_ids': torch.tensor(negative_enc.ids, dtype=torch.long),
                'attention_mask': torch.tensor(negative_enc.attention_mask, dtype=torch.long),
            }
        }

    def __len__(self):
        return len(self._dataset)


class RetrievalCollator:
    def __call__(self, items: list[dict[str, dict[str, torch.Tensor]]]):
        for i, item in enumerate(items):
            if torch.equal(item['negative']['input_ids'], item['positive']['input_ids']):
                # Ищем замену для негативного примера
                for j, replacement_item in enumerate(items):
                    if i != j and not torch.equal(replacement_item['positive']['input_ids'],
                                                  item['positive']['input_ids']):
                        item['negative'] = replacement_item['positive']
                        break
        return {
            'query': {
                'input_ids': pad_tensors([x['query']['input_ids'] for x in items]),
                'attention_mask': pad_tensors([x['query']['attention_mask'] for x in items]),
            },
            'positive': {
                'input_ids': pad_tensors([x['positive']['input_ids'] for x in items]),
                'attention_mask': pad_tensors([x['positive']['attention_mask'] for x in items]),
            },
            'negative': {
                'input_ids': pad_tensors([x['negative']['input_ids'] for x in items]),
                'attention_mask': pad_tensors([x['negative']['attention_mask'] for x in items]),
            }
        }


def create_dataloader(dataset: RetrievalDataset, collator: RetrievalCollator):
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=4,
        collate_fn=collator, pin_memory=True
    )


if __name__ == '__main__':
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    dataset = RetrievalDataset(dataset)
    collator = RetrievalCollator()
    dataloader = create_dataloader(dataset, collator)
    for batch in dataloader:
        print(batch)
        print(123)