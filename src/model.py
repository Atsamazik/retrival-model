import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from configurations.validation import load_config

config = load_config()


class RetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = AutoModel.from_pretrained(config.BASE_MODEL_NAME)
        self._query_proj = nn.Linear(self._model.config.hidden_size, self._model.config.hidden_size)
        self._document_proj = nn.Linear(self._model.config.hidden_size, self._model.config.hidden_size)

    def freeze_base_model(self):
        for param in self._model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        for param in self._model.parameters():
            param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_query: bool):
        model_outs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_out = model_outs.pooler_output
        if is_query:
            output = self._query_proj(pooled_out)
        else:
            output = self._document_proj(pooled_out)
        return F.normalize(output, dim=-1)


if __name__ == '__main__':
    model = RetrievalModel()
    result = model(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 1, 1]]), is_query=True)
    print(result)
