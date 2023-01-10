import json

from torch import nn
from transformers import DistilBertForMaskedLM, DistilBertConfig

with open("config.json") as json_file:
    config = json.load(json_file)


class NaturalnessClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        distilBertConfig = DistilBertConfig.from_json_file(config["PRE_TRAINED_CONFIG"])
        self.distilBert = DistilBertForMaskedLM.from_pretrained(config["PRE_TRAINED_MODEL"], config=distilBertConfig)

    def forward(self, input_ids, attention_mask, labels):
        output = self.distilBert(input_ids=input_ids,attention_mask=attention_mask, labels=labels).logits
        return output
