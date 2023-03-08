import json

from transformers import DistilBertForMaskedLM, DistilBertConfig

with open("config.json") as json_file:
    config = json.load(json_file)


class NaturalnessClassifier(nn.Module):
    def __init__(self):
        config = DistilBertConfig.from_json_file(config["PRE_TRAINED_CONFIG"])
        self.model = DistilBertForMaskedLM.from_pretrained(config["PRE_TRAINED_MODEL"],config=self.config)
        
    def forward(self):
        # output = self.distilBert(input_ids=input_ids,attention_mask=attention_mask, labels=labels).logits
        output = self.distilBert(input_ids=input_ids,attention_mask=attention_mask, labels=labels)
        return output
