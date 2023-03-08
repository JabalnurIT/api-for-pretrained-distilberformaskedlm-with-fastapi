import json

import torch
import math
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments
from transformers import DefaultFlowCallback

from .naturalness_classifier import NaturalnessClassifier

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config["DISTIL_BERT_MODEL"])

        self.model = NaturalnessClassifier()

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.trainer = Trainer(
            model=self.model.to(self.device),
            data_collator=self.data_collator,
        )

    def fill_mask(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=config["MAX_SEQUENCE_LEN"],
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            token_logits = self.trainer.model(input_ids, None, None).logits

        _,mask_token_index = (np.argwhere(inputs["input_ids"] == self.tokenizer.mask_token_id)).numpy()

        mask_token_logits = [token_logits[0, index, :] for index in mask_token_index]

        top_N_tokens = [np.argsort(-(logit.detach().numpy()))[:5].tolist() for logit in mask_token_logits]

        for i,mask in enumerate(top_N_tokens):
            for n,token in enumerate(mask):
                if n == 0: 
                    lastTop = self.tokenizer.decode([token])
                # print(f"mask{i}>>> {text.replace(self.tokenizer.mask_token, self.tokenizer.decode([token]), 1)}")
            text = text.replace(self.tokenizer.mask_token, lastTop, 1)
        # print(text)
        return (
            text
        )

    def ids_all(self, text):
        return list(map(lambda row: [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(row))], text))

    def dsFromList(self, text):
        return Dataset.from_pandas(pd.DataFrame(self.ids_all(text=text),columns=["input_ids"]))

    def perplexity(self, text):
        p = 0
        vnan = 0
        n = 10
        for i in range(n):
            loss = self.trainer.evaluate(self.dsFromList(text=[text]))['eval_loss']
            if math.isnan(loss):
                vnan += 1
            if not math.isnan(loss):
                p += math.exp(loss)
        perplexity = p / (n-vnan)
        return (
            perplexity
        )

    def retrain(self, text):

        id_list = self.ids_all(text=text[:])

        id_df = pd.DataFrame(id_list,columns=['input_ids'])

        dataset = Dataset.from_pandas(id_df)

        # print(dataset)
        output_dir = './assets/model/'
        training_args = TrainingArguments(
            output_dir=output_dir,
            # overwrite_output_dir=True,
            # evaluation_strategy="epoch",
            overwrite_output_dir=True,
            # evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            # per_device_train_batch_size=batch_size,
            # per_device_eval_batch_size=batch_size,
            # push_to_hub=True,
            # fp16=True,
            # logging_steps=logging_steps,
        )

        self.trainer = Trainer(
            model=self.model.to(self.device),
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator
        )

        # for cb in self.trainer.callback_handler.callbacks:
        #     if isinstance(cb, DefaultFlowCallback):
        #         self.trainer.callback_handler.remove_callback(cb)

        self.trainer.train()
        self.model.to(self.device).save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)




model = Model()


def get_model():
    return model


# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] [MASK] [MASK] [MASK],Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
# http POST http://127.0.0.1:8000/fill_mask text="1,Q42,[MASK] Adams,[MASK] writer and [MASK],[MASK],United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# http POST http://127.0.0.1:8000/perplexity text="1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"
