import torch
import torch.nn as nn
from transformers import BertTokenizer
from lstm_single_task import SingleTaskLSTM
from lstm_parameters import Args, args
from datetime import date
import os
import pandas as pd
from sklearn.metrics import f1_score

arg = Args(args)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                          max_length=12, padding="max_length")


def load_eval_dataframe(path):
    df = pd.read_excel(path)
    return df


def load_trained_dict(result_path, epoch=0, date_string="24_10_2022"):
    checkpoint_path = result_path + date_string + "_checkpoints"
    model_path = checkpoint_path + f"/{str(epoch)}/" + "cp_kidney_cancer_type_lstm.pt"
    if not os.path.isfile(model_path):
        raise Exception("Model path is incorrect")
    return torch.load(model_path)


def create_model_instance(arguments):
    instance = SingleTaskLSTM(tokenize=tokenizer,
                              emb_dim=arguments.emb_dim,
                              hid_dim=arguments.hid_dim,
                              max_len=arguments.max_seq_len,
                              bidirectional=arguments.bidirectional,
                              num_classes=2)
    return instance


def model_inference(text, model, tokenizer):
    tokenized = tokenizer([text], max_length=arg.max_seq_len, padding="max_length", truncation=True,
                          return_tensors="pt")
    out = model(tokenized["input_ids"])
    return out


def prediction(text, model, tokenizer):
    softmax = nn.Softmax()
    logits = model_inference(text, model, tokenizer)
    prob = softmax(logits)
    return logits, prob


def dataset_inference(df, model, tokenizer):
    pred2label = {0: "right",
                  1: "left"}
    final_ = []
    with torch.no_grad():
        for i in range(len(df)):
            res = {}
            patient = eval(df["Pateint_ID_text"][i])[0]
            text = df["text"][i]
            label = df["side"][i]
            logits, prob = prediction(text, model, tokenizer)

            res["patient"] = patient
            res["text"] = text
            res["true_label"] = label
            res["pred"] = pred2label[torch.argmax(prob).item()]
            res["logits"] = logits
            final_.append(res)
    return pd.DataFrame(final_)


def confusion_matrix(prediction_dataframe, true_col="true_label", pred_col="pred"):
    conf_matrix = pd.crosstab(prediction_dataframe[true_col], prediction_dataframe[pred_col], rownames=['Actual'],
                              colnames=['Predicted'])
    return conf_matrix


def get_accuracy(prediction_dataframe, true_col="true_label", pred_col="pred"):
    true_count = sum(prediction_dataframe[true_col] == prediction_dataframe[pred_col])
    return true_count / eval_predictions.shape[0]


def get_metrics(prediction_dataframe, true_col="true_label", pred_col="pred"):
    conf_mat = confusion_matrix(prediction_dataframe, true_col, pred_col)
    acc_ = get_accuracy(prediction_dataframe, true_col, pred_col)
    f1_micro = f1_score(eval_predictions[true_col], eval_predictions[pred_col], average='micro')
    f1_macro = f1_score(eval_predictions[true_col], eval_predictions[pred_col], average='macro')
    return conf_mat, acc_, f1_micro, f1_macro


eval_data = load_eval_dataframe(arg.eval_data_path)
model = create_model_instance(arg)
result_path = "results/"
model_date_string = "24_10_2022"
eval_scores = {}
os.mkdir(result_path+model_date_string+"_checkpoints/evaluation_sets")

for ep in range(10):
    epoch = ep
    trained_state_dict = load_trained_dict(result_path, epoch, model_date_string)
    model.load_state_dict(trained_state_dict)

    eval_predictions = dataset_inference(eval_data, model, model.tokenizer)
    eval_predictions.to_excel(result_path+model_date_string+"_checkpoints/evaluation_sets/"+model_date_string+f"_{ep}_th_epoch.xlsx", index=False)
    cm, acc, f1_micro, f1_macro = get_metrics(eval_predictions)
    eval_scores[ep] = {"cm": cm, "acc": acc, "f1_micro": f1_micro, "f1_macro": f1_macro}

for key in eval_scores.keys():
    print(eval_scores[key], "\n")
print("tested")
