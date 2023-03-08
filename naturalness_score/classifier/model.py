import json

import torch
import math
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments, DistilBertForMaskedLM, DistilBertConfig
from transformers import DefaultFlowCallback



with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(config["DISTIL_BERT_MODEL"])
        
        self.config = DistilBertConfig.from_json_file(config["PRE_TRAINED_CONFIG"])
        self.model = DistilBertForMaskedLM.from_pretrained(config["PRE_TRAINED_MODEL"],config=self.config)
        
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        
        self.trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
        )

    def fill_mask(self, text, N = 1):
        inputs = self.tokenizer(text, return_tensors="pt")
        token_logits = self.model(**inputs).logits
        _,mask_token_index = (np.argwhere(inputs["input_ids"] == self.tokenizer.mask_token_id)).numpy()
        mask_token_logits = [token_logits[0, index, :] for index in mask_token_index]

        top_N_tokens = [np.argsort(-(logit.detach().numpy()))[:N].tolist() for logit in mask_token_logits]
        # print(top_N_tokens)
        for i,mask in enumerate(top_N_tokens):
            for n,token in enumerate(mask):
                if(n == 0): lastTop = self.tokenizer.decode([token])
                # print(f"mask{i}>>> {text.replace(self.tokenizer.mask_token, self.tokenizer.decode([token]), 1)}")
            text = text.replace(self.tokenizer.mask_token, lastTop, 1)
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
        
        output_dir = config["OUTPUT_DIR"]
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=2e-5,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator
        )

        # for cb in self.trainer.callback_handler.callbacks:
        #     if isinstance(cb, DefaultFlowCallback):
        #         self.trainer.callback_handler.remove_callback(cb)
        try:
            trainer.train()
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            status = "Success"
        except:
            status = "Failed"
        return status
        
        




model = Model()


def get_model():
    return model

# fill_mask
# http POST http://127.0.0.1:8000/fill_mask text="Q23,George [MASK],1st [MASK] of the United States (1732–1799),[MASK],United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0"
# http POST http://127.0.0.1:8000/fill_mask text="Q42,[MASK] Adams,[MASK] writer and [MASK],[MASK],United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# perplexity
# http POST http://127.0.0.1:8000/perplexity text="Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0"

# retrain 
# http POST http://127.0.0.1:8000/retrain text="Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0"

# ['Q23,George Washington,1st president of the United States (1732–1799),Male,United States of America; Kingdom of Great Britain,Politician,1732,1799.0,natural causes,67.0','Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0','Q91,Abraham Lincoln,16th president of the United States (1809-1865),Male,United States of America,Politician,1809,1865.0,homicide,56.0','Q254,Wolfgang Amadeus Mozart,Austrian composer of the Classical period,Male,Archduchy of Austria; Archbishopric of Salzburg,Artist,1756,1791.0,,35.0']