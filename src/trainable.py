from pathlib import Path

import torch
from datasets import load_from_disk
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import ContrastiveLoss

from configurations.validation import load_config
from src.dataset import RetrievalDataset, RetrievalCollator
from src.model import RetrievalModel

config = load_config()
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


class RetrievalTrainable:
    def __init__(self, model: RetrievalModel):
        self._model = model
        self.losses = {
            "contrastive_loss": ContrastiveLoss(pos_margin=0.1,
                                                neg_margin=0.9,
                                                distance=CosineSimilarity()),
            "triplet_margin_loss":  torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        }

        self._loss = self.losses[config.loss.value]


    def compute_loss(self, data: dict):
        def compute_contrastive_loss():
            query_embeds = self._model(data['query']['input_ids'], data['query']['attention_mask'], is_query=True)
            document_embeds = self._model(data['positive']['input_ids'], data['positive']['attention_mask'], is_query=False)
            all_embeds = torch.cat((query_embeds, document_embeds), dim=0)
            partial_labels = torch.arange(0, len(query_embeds), device=query_embeds.device, dtype=torch.long)
            all_labels = torch.cat((partial_labels, partial_labels), dim=0)
            loss_value = self._loss(all_embeds, all_labels)
            return loss_value

        def compute_triplet_margin_loss():
            query_embeds = self._model(data['query']['input_ids'], data['query']['attention_mask'],
                                       is_query=True)
            positive_embeds = self._model(data['positive']['input_ids'], data['positive']['attention_mask'],
                                          is_query=False)
            negative_embeds = self._model(data['negative']['input_ids'], data['negative']['attention_mask'],
                                          is_query=False)
            loss_value = self._loss(query_embeds, positive_embeds, negative_embeds)
            return loss_value

        if config.loss.value == "contrastive_loss":
            return compute_contrastive_loss()
        if config.loss.value == "triplet_margin_loss":
            return compute_triplet_margin_loss()

if __name__ == '__main__':
    data = load_from_disk(PROCESSED_DATA_DIR)
    dataset = RetrievalDataset(data)
    collator = RetrievalCollator()
    model = RetrievalModel()
    batch = collator([dataset[0], dataset[1], dataset[2], dataset[3]])
    trainable = RetrievalTrainable(model)
    loss = trainable.compute_loss(batch)
    print(loss)
