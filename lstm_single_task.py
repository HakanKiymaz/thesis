import numpy as np
import torch.nn as nn
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from lstm_parameters import Args, args
from helpers import create_kidney_label, preprocess

arguments = Args(args, type="kidney")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SingleTaskLSTM(nn.Module):
    def __init__(self, arg_class, num_classes, vocab_size):
        super(SingleTaskLSTM, self).__init__()
        self.D = 2 if arg_class.bidirectional else 1

        emb_size = vocab_size
        self.emb_dim = arg_class.emb_dim

        self.embed = nn.Embedding(emb_size, arg_class.emb_dim)
        self.lstm = nn.LSTM(input_size=arg_class.emb_dim, hidden_size=arg_class.hid_dim,
                            bidirectional=arg_class.bidirectional, batch_first=True)
        self.fc1 = nn.Linear(arg_class.hid_dim * self.D, num_classes)
        self.drop = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        self.hidden_dim = arg_class.hid_dim
        self.cell_dim = arg_class.hid_dim
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, batch_mask):
        current_batch_size = len(x)
        embed = self.embed(x)  # Get embedding features of input ids

        h0 = self.init_hidden(current_batch_size)
        c0 = self.init_cell(current_batch_size)
        lstm_out, (hidden, cell) = self.lstm(embed, (
            h0, c0))  # lstm takes input features matrix and (hidden,cell) state vectors
        # mask filtered embeddings are fed into lstm

        cell = cell.permute((1, 0, 2))
        cell = torch.flatten(cell, 1)

        output = self.drop(cell)
        output = self.activation(output)
        output = self.last_token_vector(output, batch_mask)
        output = self.fc1(output)  # fully connected projection layer for class logits

        return output

    def init_hidden(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False, device='cuda:0')

    def init_cell(self, batch_size):
        return torch.zeros(self.D, batch_size, self.hidden_dim, requires_grad=False, device='cuda:0')

    @staticmethod
    def last_token_vector(output_, mask_):
        last_tokens = mask_.sum(dim=1) - 1
        return torch.stack([o[last_tokens[i]] for i, o in enumerate(output_)])


class PathologyDataset(Dataset):
    def __init__(self, dataframe, max_len, tokenizer):
        super(PathologyDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.texts = self.tokenizer(self.data.text.values.tolist(), max_length=max_len, padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")
        """
        We use Bert tokenizer from huggingface.
        Tokenizer gives 3 outputs which are pytorch tensors. 
        (input_ids, token_type_ids, attention_mask)
        """
        self.input_ids = self.texts["input_ids"]
        self.token_type_ids = self.texts["token_type_ids"]
        self.attention_mask = self.texts["attention_mask"]  # pad

        self.labels_ = self.data.side.values.tolist()
        self.label_map = {"right": torch.Tensor([1, 0]),
                          "left": torch.Tensor([0, 1])}
        self.labels = [self.label_map[lbl] for lbl in self.labels_]
        self.labels = torch.stack(self.labels)

    def __getitem__(self, idx):
        # this method will be called when we iterate dataset in the for loop.
        text_ = self.input_ids[idx]
        labels_ = self.labels[idx]
        mask_ = self.attention_mask[idx]
        return text_, labels_, mask_

    def __len__(self):
        return len(self.data)


def save_model(instance, path_to_save):
    torch.save(instance.state_dict(), path_to_save)


if __name__ == "__main__":
    # result_path = "results/"
    # day_string = date.today().strftime("%Y_%m_%d")
    # checkpoint_path = result_path + day_string + "_checkpoints"
    # if os.path.isdir(checkpoint_path):
    #     raise Exception("Model Path Already Exist")
    #
    # os.mkdir(checkpoint_path)

    tabular_data = pd.read_excel(arguments.data_path,
                                 usecols=["Pateint_ID_text", "TEXT", "Anatomical_Position_type"])

    tabular_data["side"] = tabular_data["Anatomical_Position_type"].apply(create_kidney_label)

    tabular_data = tabular_data[tabular_data["side"] != "None"]
    tabular_data["text"] = tabular_data["TEXT"].apply(preprocess)

    tabular_data.drop(["Anatomical_Position_type", "TEXT"], axis=1, inplace=True)
    train_data, eval_data = train_test_split(tabular_data, test_size=0.2, random_state=42)

    # train_data.to_excel(arguments.train_data_path, index=False)
    # eval_data.to_excel(arguments.eval_data_path, index=False)
    train_data.drop("Pateint_ID_text", axis=1, inplace=True)
    eval_data.drop("Pateint_ID_text", axis=1, inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    eval_data.reset_index(drop=True, inplace=True)

    tokenizer = BertTokenizer.from_pretrained(arguments.bert_model_path,
                                              max_length=arguments.tokenizer_max_len, padding="max_length")

    train_dataset = PathologyDataset(train_data, max_len=arguments.max_seq_len, tokenizer=tokenizer)
    eval_dataset = PathologyDataset(eval_data, max_len=arguments.max_seq_len, tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=True)

    model = SingleTaskLSTM(arguments,
                           num_classes=2,
                           vocab_size=len(tokenizer.vocab))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    losses = {"train_loss": [], "eval_loss": []}
    for epoch in range(arguments.epochs):
        train_loss = []
        eval_loss = []
        # os.mkdir(checkpoint_path + f"/{str(epoch)}")
        model.train()
        for features, labels, batch_mask in train_dataloader:
            features = features.to(device)
            labels = labels.to(device)
            batch_mask = batch_mask.to(device)
            if features.shape[0] == 1:
                labels = labels
            else:
                labels = labels.squeeze()
            optimizer.zero_grad()

            out = model(features, batch_mask)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            losses["train_loss"].append(np.mean(train_loss))
        model.eval()
        with torch.no_grad():
            for features, labels, batch_mask in eval_dataloader:
                features = features.to(device)
                labels = labels.to(device)
                batch_mask = batch_mask.to(device)
                if features.shape[0] == 1:
                    labels = labels
                else:
                    labels = labels.squeeze()
                out = model(features, batch_mask)
                loss = criterion(out, labels)
                eval_loss.append(loss.item())
                losses["eval_loss"].append(np.mean(eval_loss))
        print()
        # save_model(model, checkpoint_path + f"/{str(epoch)}/" + "cp_kidney_cancer_type_lstm.pt")
        # with open(checkpoint_path + "/train_loss.txt", "w") as f:
        #     f.write(str(np.mean(train_loss)) + "\n")
        # with open(checkpoint_path + "/eval_loss.txt", "w") as f:
        #     f.write(str(np.mean(eval_loss)) + "\n")
