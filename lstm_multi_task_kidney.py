import os
from collections import OrderedDict

import numpy as np
from datetime import date
import torch.nn as nn
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from lstm_parameters import Args, args
from helpers import preprocess

arguments = Args(args, type="kidney")


class MultiTaskKidneyLSTM(nn.Module):
    def __init__(self, arg_class, task_count=3, task_num_classes=None):
        super(MultiTaskKidneyLSTM, self).__init__()
        if task_num_classes is None:
            task_count = 3
            task_num_classes = [3, 4, 5]

        self.D = 2 if arg_class.bidirectional else 1
        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/hkiymaz/.cache/huggingface/transformers/45c3f7a79a80e1cf0a489e5c62b43f173c15db47864303a55d623bb3c96f72a5.d789d64ebfe299b0e416afc4a169632f903f693095b4629a7ea271d5a0cf2c99",
            max_length=arg_class.tokenizer_max_len, padding="max_length",
            local_files_only=True)
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

        output = self.drop(hidden)
        output = self.activation(output)
        task_outputs = [layer(output) for layer in self.output_layers]
        return task_outputs

    def init_hidden(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False)

    def init_cell(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False)


class KidneyMultiTaskDataset(Dataset):
    def __init__(self, dataframe, max_len, tokenize=None):
        super(KidneyMultiTaskDataset, self).__init__()
        self.data = dataframe
        self.max_len = max_len
        self.texts = tokenize(self.data.text.values.tolist(), max_length=max_len, padding="max_length",
                              truncation=True,
                              return_tensors="pt")
        self.input_ids = self.texts["input_ids"]
        self.token_type_ids = self.texts["token_type_ids"]
        self.attention_mask = self.texts["attention_mask"]
        self.labels = self.data.anatomical_side.values.tolist()
        self.label_map = {"right": torch.Tensor([1, 0]),
                          "left": torch.Tensor([0, 1])}
        print("label_map", self.label_map)

    def __getitem__(self, idx):
        batch_text = [text for text in self.input_ids][idx]
        batch_label = [self.label_map[lbl] for lbl in self.labels][idx]
        return batch_text, batch_label

    def __len__(self):
        return len(self.data)


def save_model(instance, path_to_save):
    torch.save(instance.state_dict(), path_to_save)


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


class KidneyDataClean:
    def __init__(self, ):
        pass

    @staticmethod
    def adjust_ajcc(row: list):
        row = eval(row)
        row = list(set(row))
        if len(row) > 1:
            return "AJ1"
        elif None in row:
            return "None"
        else:
            return "AJ0"

    @staticmethod
    def adjust_histologic_type(row: list):
        row = eval(row)
        row = list(set(row))
        if len(row) > 1:
            return "multi_type"
        elif None in row:
            return "None"
        elif len(row) == 1:
            if "other" in row:
                return "other"
            elif "Clear_Cell_Renal_Carcinoma" in row:
                return "clear_cell_renal_carcinoma"
            elif "Papillary_Renal_Cell_Carcinoma" in row:
                return "papillary_renal_cell_carcinoma"
            else:
                return "None"

    @staticmethod
    def create_kidney_label(text):
        if "left" in text.lower():
            return "left"
        elif "right" in text.lower():
            return "right"
        else:
            return "None"


if __name__ == "__main__":
    targets = arguments.MT_target_cols

    kdc = KidneyDataClean()

    model_data = pd.read_excel(arguments.data_path,
                               usecols=arguments.MT_other_cols + arguments.MT_target_cols)

    model_data["anatomical_side"] = model_data["Anatomical_Position_type"].apply(kdc.create_kidney_label)
    model_data["ajcc_classification"] = model_data["AJCC_Classification_id"].apply(kdc.adjust_ajcc)
    model_data["histologic_classification"] = model_data["Histologic_Type_type"].apply(kdc.adjust_histologic_type)

    model_data = model_data[model_data["anatomical_side"] != "None"]
    model_data["text"] = model_data["TEXT"].apply(preprocess)

    model_data.drop(["Anatomical_Position_type", "TEXT"], axis=1, inplace=True)
    train_data, eval_data = train_test_split(model_data, test_size=0.2, random_state=42)

    # train_data.to_excel(arguments.train_data_path, index=False)
    # eval_data.to_excel(arguments.eval_data_path, index=False)
    task = TaskCreator(model_data, targets)
    model = MultiTaskKidneyLSTM(arguments,
                                task.task_count, task.task_num_classes)

    train_data.drop("Pateint_ID_text", axis=1, inplace=True)
    eval_data.drop("Pateint_ID_text", axis=1, inplace=True)
    train_dataset = KidneyMultiTaskDataset(train_data, max_len=arguments.max_seq_len, tokenize=model.tokenizer)
    eval_dataset = KidneyMultiTaskDataset(eval_data, max_len=arguments.max_seq_len, tokenize=model.tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    losses = {"train_loss": [], "eval_loss": []}
    for epoch in range(arguments.epochs):
        train_loss = []
        eval_loss = []
        model.train()
        for features, labels in train_dataloader:
            optimizer.zero_grad()

            out = model(features)
            out = model.softmax(out)
            loss = criterion(out, labels.squeeze())

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        losses["train_loss"].append(np.mean(train_loss))
        model.eval()
        with torch.no_grad():
            for features, labels in eval_dataloader:
                out = model(features)
                loss = criterion(out, labels.squeeze())
                eval_loss.append(loss.item())
        losses["eval_loss"].append(np.mean(eval_loss))

        # save_model(model, checkpoint_path + f"/{str(epoch)}/" + "cp_kidney_cancer_type_lstm.pt")
        # with open(checkpoint_path + "/train_loss.txt", "w") as f:
        #     f.write(str(np.mean(train_loss)) + "\n")
        # with open(checkpoint_path + "/eval_loss.txt", "w") as f:
        #     f.write(str(np.mean(eval_loss)) + "\n")
