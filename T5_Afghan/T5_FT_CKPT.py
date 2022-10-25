import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, \
    AdamW

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name) # using t5 since distilbert dont work well with the complex question - answer pipeline

with open('/content/Voice_Chatbot/T5_Afghan/context_afghan.json', encoding='utf8') as json_file:
    data = json.load(json_file)

def extract_questions_and_answers(path):
    with open(path, encoding='utf8') as json_file:
        data = json.load(json_file)
    questions = data["materiel"]
    data_row = []
    for question in questions:
        context = question["context"]
        for question_and_answers in question["qas"]:
            q = question_and_answers["question"]
            a = question_and_answers["answers"]

            for answer in a:
                answer_text = answer["text"]
                answer_start = answer["answer_start"]
                answer_end = answer_start + len(answer_text) # get the end of the answer so the model knows what's going on

                data_row.append({
                    "question" : q,
                    "context" : context,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                    "answer_end":answer_end
                })
    return pd.DataFrame(data_row) # make a data frome that has all the info in it


df = extract_questions_and_answers('/content/Voice_Chatbot/T5_Afghan/context_afghan.json')

class dataset_creation(Dataset):
    def __init__(self, data, tokenizer, source_max_token_len = 396,
                 target_max_token_length = 64):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_length = target_max_token_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index:int):
        data_row = self.data.iloc[index]

        source_encoding = tokenizer(
            data_row['question'],
            data_row['context'][0], # put zero since the data_row["context"] is a list and we want just the first string of the list
            max_length=self.source_max_token_len, # max length is being given so that all the sentences can be of the same length - important to pad for training purposes - since the model takes a fixed size
            padding="max_length",  # padding equal to the max_length
            truncation="only_second",  # only truncate the context - since we only want it till the answer or upto the answer
            return_attention_mask=True, # attention mask can be fed to the model for better efficiency during training - this masks helps find out the weightage of a word wrt a sentence that came before or the context of the question
            add_special_tokens=True, # research on it!!!!!!
            return_tensors="pt" # return the tensors in a pytorch format
        )

        target_encoding = tokenizer(
            data_row['answer_text'],
            max_length=self.target_max_token_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            question = data_row["question"],
            context = data_row["context"],
            answer_text = data_row["answer_text"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask = source_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )

sample_dataset = dataset_creation(df, tokenizer)

'''
for d in sample_dataset:
    print(d["question"])
    print(d["answer_text"])
    print(d["input_ids"][:20])
    print(d["labels"][:20])
'''


###TESTING!!!###
#pd.set_option('max_columns', 5)
#print(df)
#print(df['context'][0])

train_df, val_df = train_test_split(df, test_size=0.05)
print(train_df.shape, val_df.shape)

# SET THE NUMBER OF WORKERS HERE

class data_module(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size = 1,
                 source_max_token_len=396,
                 target_max_token_length=64
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_length = target_max_token_length

    def setup(self, stage=None):
        self.train_dataset = dataset_creation(
            self.train_df, self.tokenizer, self.source_max_token_len, self.target_max_token_length
        )

        self.test_dataset = dataset_creation(
            self.test_df, self.tokenizer, self.source_max_token_len, self.target_max_token_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=2
        )

BATCH_SIZE = 2 # BATCH SIZE
N_EPOCH = 40 # N_EPOCHS

data_module_use = data_module(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
data_module_use.setup()

#model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict = True)

class QA_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def val_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.0001)

trained_model = QA_Model.load_from_checkpoint("last_afghan.ckpt")
trained_model.freeze()

def generate_answer(question):

  source_encoding = tokenizer(
            question['question'],
            question['context'][0], # put zero since the data_row["context"] is a list and we want just the first string of the list
            max_length=1024, # max length is being given so that all the sentences can be of the same length - important to pad for training purposes - since the model takes a fixed size
            padding="max_length",  # padding equal to the max_length
            truncation="only_second",  # only truncate the context - since we only want it till the answer or upto the answer
            return_attention_mask=True, # attention mask can be fed to the model for better efficiency during training - this masks helps find out the weightage of a word wrt a sentence that came before or the context of the question
            add_special_tokens=True, # research on it!!!!!!
            return_tensors="pt" # return the tensors in a pytorch format
        )
  generated_ids = trained_model.model.generate(
          input_ids = source_encoding["input_ids"],
          attention_mask = source_encoding["attention_mask"],
          num_beams = 4, # how many beam searches you want to have
          max_length = 100,
          repetition_penalty = 2.5,
          length_penalty = 1.0,
          early_stopping=True,
          use_cache=True
      )

  pred = [
          tokenizer.decode(generated_id, skip_special_tokens=True,
                           clean_up_tokenization_spaces=True)
          for generated_id in generated_ids
  ]

  return "".join(pred)

# custom function for the generation of the answers on the custom questions from the user
def generate_answer_custom(question):

  source_encoding = tokenizer(
            question,
            train_df['context'][0][0], # put zero since the data_row["context"] is a list and we want just the first string of the list
            max_length=512, # max length is being given so that all the sentences can be of the same length - important to pad for training purposes - since the model takes a fixed size
            padding="max_length",  # padding equal to the max_length
            truncation="only_second",  # only truncate the context - since we only want it till the answer or upto the answer
            return_attention_mask=True, # attention mask can be fed to the model for better efficiency during training - this masks helps find out the weightage of a word wrt a sentence that came before or the context of the question
            add_special_tokens=True, # research on it!!!!!!
            return_tensors="pt" # return the tensors in a pytorch format
        )
  generated_ids = trained_model.model.generate(
          input_ids = source_encoding["input_ids"],
          attention_mask = source_encoding["attention_mask"],
          num_beams = 3, # greedy search
          max_length = 300,
          repetition_penalty = 2.5,
          length_penalty = 1.65,
          early_stopping=True,
          use_cache=True
      )

  pred = [
          tokenizer.decode(generated_id, skip_special_tokens=True,
                           clean_up_tokenization_spaces=True)
          for generated_id in generated_ids
  ]

  return "".join(pred)

#sample_question_1 = 'How do you live on a small pay?'
#print(generate_answer_custom(sample_question_1))

