import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
import torch
import pandas as pd
import os
import numpy as np
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from datetime import date

from lstm_parameters import Args, args
from helpers import create_kidney_label, preprocess, get_existing_reports, COLUMNS

arguments = Args(args, type="kidney")


class MultiTaskLSTM(nn.Module):
    def __init__(self, arg_class, task_count=3, task_num_classes=None):
        super(MultiTaskLSTM, self).__init__()
        if task_num_classes is None:
            task_count = 3
            task_num_classes = [3, 4, 5]

        self.D = 2 if arg_class.bidirectional else 1
        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/hkiymaz/.cache/huggingface/transformers/45c3f7a79a80e1cf0a489e5c62b43f173c15db47864303a55d623bb3c96f72a5.d789d64ebfe299b0e416afc4a169632f903f693095b4629a7ea271d5a0cf2c99",
            max_length=arg_class.tokenizer_max_len, padding="max_length",
            local_files_only=True)
        # self.tokenizer = BertTokenizer.from_pretrained(arg_class.bert_model_path,
        #                                                max_length=arg_class.tokenizer_max_len, padding="max_length",
        #                                                cache_dir="PATH/TO/MY/CACHE/DIR")
        emb_size = len(self.tokenizer.vocab)
        self.emb_dim = arg_class.emb_dim

        self.embed = nn.Embedding(emb_size, arg_class.emb_dim)
        self.lstm = nn.LSTM(input_size=arg_class.emb_dim, hidden_size=arg_class.hid_dim,
                            bidirectional=arg_class.bidirectional, batch_first=True)

        self.drop = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.hidden_dim = arg_class.hid_dim
        self.cell_dim = arg_class.hid_dim
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.output_layers = nn.ModuleList(
            [nn.Linear(arg_class.hid_dim, task_num_classes[i]) for i in range(task_count)])

    def forward(self, x):
        current_batch_size = len(x)
        embed = self.embed(x)
        h0 = self.init_hidden(current_batch_size)
        c0 = self.init_cell(current_batch_size)
        lstm_out, (hidden, cell) = self.lstm(embed, (h0, c0))
        """
        the output of hidden cell is D*num_layers,N,Hidden_out.
        """
        output = self.drop(hidden)
        output = self.activation(output)
        task_outputs = [layer(output) for layer in self.output_layers]
        return task_outputs

    def init_hidden(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False)

    def init_cell(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False)


class PathologyMultiClassDataset(Dataset):
    def __init__(self, features, labels, max_len, tokenize=None):
        super(PathologyMultiClassDataset, self).__init__()

        self.features = features
        self.labels = labels

        self.max_len = max_len
        self.texts = tokenize(self.features, max_length=max_len, padding="max_length",
                              truncation=True,
                              return_tensors="pt")
        self.input_ids = self.texts["input_ids"]
        self.token_type_ids = self.texts["token_type_ids"]
        self.attention_mask = self.texts["attention_mask"]

    def __getitem__(self, idx):
        batch_text = [text for text in self.input_ids][idx]
        batch_label = [self.labels[lbl][idx] for lbl in self.labels]
        return batch_text, batch_label

    def __len__(self):
        return len(self.features)


class TaskCreator:
    def __init__(self, dataframe, targets=None):
        self.data = dataframe
        self.targets = targets
        if not targets:
            self.targets = list(set(dataframe.columns.to_list()) - {"bcr_patient_barcode", "text"})

        self.task_count = len(self.targets)
        self.task_num_classes = []

        self.labels = OrderedDict()
        self.create_index()

        self.features = self.data.text.values.tolist()
        for lbl in self.targets:
            tmp_dct = self.expand_labels_to_vectors(self.c2i[lbl])
            self.labels[lbl] = [tmp_dct[val] for val in self.data[lbl].values.tolist()]

    def expand_labels_to_vectors(self, label_dict):
        mt = np.diag(np.ones(len(label_dict)))
        return {keyy: mt[ix] for ix, keyy in enumerate(label_dict.keys())}

    def create_index(self):
        self.label_to_index()
        self.index_to_label()
        self.class_to_index()

    def label_to_index(self):
        self.l2i = {l: idx for idx, l in enumerate(self.targets)}

    def index_to_label(self):
        self.i2l = {idx: l for idx, l in enumerate(self.targets)}

    def class_to_index(self):
        self.c2i = {}
        self.i2c = {}
        for lbl_ in self.targets:
            uniqs = list(self.data[lbl_].unique())
            uniqs.sort(reverse=True)
            if uniqs[0] != "None" and "None" in uniqs:
                uniqs.remove("None")
                uniqs.insert(0, "None")
            # elif "None" not in uniqs:
                # uniqs.insert(0, "None")
            self.c2i[lbl_] = {}
            self.i2c[lbl_] = {}
            for ix, uniq in enumerate(uniqs):
                self.c2i[lbl_][uniq] = ix
                self.i2c[lbl_][ix] = uniq

            self.task_num_classes.append(len(uniqs))


def clean_labels(data):
    data["histological_type"].replace("Other, specify", "Others", inplace=True)
    data["histological_type"].replace("Mixed Histology (please specify)", "Mixed Histology", inplace=True)
    data["ajcc_pathologic_tumor_stage"].replace("[Not Available]", "Not Available", inplace=True)
    data["ajcc_pathologic_tumor_stage"].replace("[Discrepancy]", "Discrepancy", inplace=True)
    return data


if __name__ == "__main__":
    is_data_pre = True
    targets = ["histological_type", "tumor_status",
               "ajcc_pathologic_tumor_stage", "new_tumor_event_type"]

    if not is_data_pre:
        model_data = get_existing_reports()
        model_data = clean_labels(model_data)

        train_data, val_ = train_test_split(model_data, test_size=0.3, random_state=123)
        valid_data, test_data = train_test_split(val_, test_size=0.66, random_state=123)
        model_data.to_excel("2023-04-03_breast_full.xlsx", index=False)
        train_data.to_excel("2023-04-03_breast_train.xlsx", index=False)
        valid_data.to_excel("2023-04-03_breast_valid.xlsx", index=False)
        test_data.to_excel("2023-04-03_breast_test.xlsx", index=False)
    else:
        model_data = pd.read_excel("2023-04-03_breast_full.xlsx")
        train_data = pd.read_excel("2023-04-03_breast_train.xlsx", nrows=100)
        valid_data = pd.read_excel("2023-04-03_breast_valid.xlsx", nrows=40)
        test_data = pd.read_excel("2023-04-03_breast_test.xlsx", nrows=40)

    #model_data = pd.read_excel("12-03-2023_model_data.xlsx")
    task = TaskCreator(model_data, targets)

    model = MultiTaskLSTM(arguments, task.task_count, task.task_num_classes)
    train_dataset = PathologyMultiClassDataset(task.features, task.labels, max_len=arguments.max_seq_len, tokenize=model.tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # train-valid-test seti hazırlanacak. (70-10-20)
    # categorical cross-entropy loss olmalı. class'lar disjoint olduğu için.

    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for features, labels in train_dataloader:
        """
        {0: 'histological_type', 8 classes
         1: 'tumor_status', 3 classes
         2: 'ajcc_pathologic_tumor_stage', 14 classes
         3: 'new_tumor_event_type', 5 classes
         }
         """
        optimizer.zero_grad()

        out = model(features)
        """
        out[i].shape = D x Batch x Class
        """
        loss = torch.Tensor([0])
        print("output fc1 weights sum before backward")
        print(model.output_layers[0].weight.sum())
        for i in range(len(out)):
            individual_class_out = torch.sum(out[i], dim=0)
            loss += criterion(individual_class_out, labels[i])

        loss.backward()
        optimizer.step()

        print("output fc1 weights sum after step")
        print(model.output_layers[0].weight.sum())
        break