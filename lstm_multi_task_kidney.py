import os
import json
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

        self.drop = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        self.hidden_dim = arg_class.hid_dim
        self.cell_dim = arg_class.hid_dim
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.output_layers = nn.ModuleList(
            [nn.Linear(arg_class.hid_dim, task_num_classes[i]) for i in range(task_count)])

    def forward(self, inputs, mask):
        current_batch_size = len(inputs)
        embed = self.embed(inputs)
        h0 = self.init_hidden(current_batch_size)
        c0 = self.init_cell(current_batch_size)
        lstm_out, (hidden, cell) = self.lstm(embed, (h0, c0))

        output = self.drop(hidden)
        output = self.activation(output)
        output = torch.sum(output, dim=0)
        task_outputs = [layer(output) for layer in self.output_layers]
        return task_outputs

    def init_hidden(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False)

    def init_cell(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False)


class KidneyMultiTaskDataset(Dataset):
    def __init__(self, dataframe, task_labels, max_len, tokenize=None):
        super(KidneyMultiTaskDataset, self).__init__()
        self.data = dataframe
        self.max_len = max_len
        self.texts = tokenize(self.data.text.values.tolist(), max_length=max_len, padding="max_length",
                              truncation=True,
                              return_tensors="pt")
        self.input_ids = self.texts["input_ids"]
        self.token_type_ids = self.texts["token_type_ids"]
        self.attention_mask = self.texts["attention_mask"]
        self.labels = task_labels

    def __getitem__(self, idx):
        batch_text = [text for text in self.input_ids][idx]
        batch_mask = [text for text in self.attention_mask][idx]
        batch_label = []
        for key in self.labels.keys():
            batch_label.append(self.labels[key][idx])
        return batch_text, batch_label, batch_mask

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
        for target in self.targets:
            uniqs = list(self.data[target].unique())
            if None in uniqs:
                uniqs.remove(None)
            uniqs.sort(reverse=True)
            if "None" in uniqs:
                print(target)
                uniqs.remove("None")
                uniqs.insert(0, None)
            if None not in uniqs:
                uniqs.insert(0, None)
            self.c2i[target] = {}
            self.i2c[target] = {}
            for ix, uniq in enumerate(uniqs):
                self.c2i[target][uniq] = ix
                self.i2c[target][ix] = uniq

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
            return None
        else:
            return "AJ0"

    @staticmethod
    def adjust_histologic_type(row: list):
        row = eval(row)
        row = list(set(row))
        if len(row) > 1:
            return "multi_type"
        elif None in row:
            return None
        elif len(row) == 1:
            if "other" in row:
                return "other"
            elif "Clear_Cell_Renal_Carcinoma" in row:
                return "clear_cell_renal_carcinoma"
            elif "Papillary_Renal_Cell_Carcinoma" in row:
                return "papillary_renal_cell_carcinoma"
            else:
                return None

    @staticmethod
    def adjust_fuhrman(row):
        row = eval(row)
        row = list(set(row))
        if len(row) > 1:
            """
            if multi grade, choose max
            """
            row.sort()
            return row[-1]
        else:
            return row[0]

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
    model_data["fuhrman_type"] = model_data["Fuhrman_Nuclear_Grade_type"].apply(kdc.adjust_fuhrman)

    model_data = model_data[model_data["anatomical_side"] != "None"]
    model_data["text"] = model_data["TEXT"].apply(preprocess)

    model_data.drop(arguments.MT_target_cols + ["TEXT"], axis=1, inplace=True)
    train_data, eval_data = train_test_split(model_data, test_size=0.2, random_state=42)

    # train_data.to_excel(arguments.train_data_path, index=False)
    # eval_data.to_excel(arguments.eval_data_path, index=False)
    task = TaskCreator(model_data, arguments.MT_renamed_target_cols)
    model = MultiTaskKidneyLSTM(arguments,
                                task.task_count, task.task_num_classes)

    train_data.drop("Pateint_ID_text", axis=1, inplace=True)
    eval_data.drop("Pateint_ID_text", axis=1, inplace=True)
    train_dataset = KidneyMultiTaskDataset(train_data, task.labels, max_len=arguments.max_seq_len, tokenize=model.tokenizer)
    eval_dataset = KidneyMultiTaskDataset(eval_data, task.labels, max_len=arguments.max_seq_len, tokenize=model.tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=arguments.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=arguments.test_batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    accuracy = {"train_acc": [], "eval_acc": []}
    losses = {"train_loss": [], "eval_loss": []}
    for epoch in range(arguments.epochs):
        true_count_for_acc = 0
        total_count_for_acc = 0
        train_loss = []
        eval_loss = []
        model.train()
        for features, labels, mask in train_dataloader:
            total_count_for_acc += task.task_count * len(features)
            optimizer.zero_grad()

            out = model(features, mask)
            """
            bidirectional x batch x class_labels
            """
            loss = torch.Tensor([0.0])
            for cls_id in range(len(labels)):
                loss += criterion(out[cls_id], labels[cls_id])
                true_count_for_acc += torch.sum(out[cls_id].argmax(dim=1) == labels[cls_id].argmax(dim=1)).item()
            loss = loss / task.task_count

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        losses["train_loss"].append(np.mean(train_loss))
        accuracy["train_acc"].append(true_count_for_acc / total_count_for_acc)

        model.eval()
        with torch.no_grad():
            for features, labels, mask_ in eval_dataloader:
                out = model(features, mask_)
                loss = torch.Tensor([0.0])
                for cls_id in range(len(labels)):
                    loss += criterion(out[cls_id], labels[cls_id])
                    true_count_for_acc += torch.sum(out[cls_id].argmax(dim=1) == labels[cls_id].argmax(dim=1)).item()

                loss = loss / task.task_count
                eval_loss.append(loss.item())
        losses["eval_loss"].append(np.mean(eval_loss))
        accuracy["eval_acc"].append(true_count_for_acc / total_count_for_acc)

    print()
    with open("acc.json", "w") as f:
        f.write(json.dumps(accuracy, indent=4))
    with open("loss.json", "w") as f:
        f.write(json.dumps(losses, indent=4))
    print("accuracy:\n", accuracy, "\n")
    print("losses:\n", losses, "\n")
