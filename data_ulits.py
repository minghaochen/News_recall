import os
import random
from turtle import pos
import torch.utils.data as data
from transformers import AutoTokenizer, BertTokenizer
from tqdm import tqdm

class Supervised(data.Dataset):
    def __init__(self) -> None:
        super(Supervised, self).__init__()
        self.split_char = "|||"
        self.doc = {}
        self.train_data = {}
        self._create_train_data()
        self.tokenizer = BertTokenizer.from_pretrained('./tmp/test-mlm/checkpoint-10000')
        self.doc_keys = list(self.doc.keys())
        self.num_doc = len(self.doc)

    def _create_train_data(self):
        count = 0
        with open("data/News2022_doc.tsv", "r", encoding="utf8") as fr:
            for line in tqdm(fr):
                if count == 0:
                    count += 1
                    continue
                text = line.strip().split("\t")
                self.doc[text[0]] = text[1]
        
        line_count = 0
        for name in ["train", "dev"]:
            count = 0
            with open("data/News2022_{}.tsv".format(name), "r", encoding="utf8") as fr:
                for line in tqdm(fr):
                    if count == 0:
                        count += 1
                        continue
                    text = line.strip().split("\t")
                    self.train_data[line_count] = text[2] + self.split_char + self.doc[text[1]]
                    line_count += 1

    def __getitem__(self, index):
        anchor_text, pos_text = self.train_data[index].split(self.split_char)
        # 考虑增加负采样       
        tmp = random.randint(0, self.num_doc)
        neg_text = self.doc[self.doc_keys[tmp]]
        sample = self._post_process(anchor_text, pos_text, neg_text)
        return sample

    def _post_process(self, anchor_text, pos_text, neg_text):
        sample = self.tokenizer([anchor_text, pos_text, neg_text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=256,
                                padding='max_length',
                                return_tensors='pt')

        return sample

    def __len__(self):
        return len(self.train_data)