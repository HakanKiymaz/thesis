args = {
    "max_seq_len": 256,
    "emb_dim": 200,
    "hid_dim": 128,
    "bidirectional": True,
    "epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "data_path": "PathologyDataset/structured_sampledataset.xlsx",
    "train_data_path": "PathologyDataset/structured_sampledataset_train.xlsx",
    "eval_data_path": "PathologyDataset/structured_sampledataset_eval.xlsx",
    "bert_model_path": "bert-base-uncased",
    "tokenizer_max_len": 12
}


class Args(object):
    def __init__(self, arg_list):
        self.max_seq_len = arg_list["max_seq_len"]
        self.emb_dim = arg_list["emb_dim"]
        self.hid_dim = arg_list["hid_dim"]
        self.bidirectional = arg_list["bidirectional"]
        self.epochs = arg_list["epochs"]
        self.learning_rate = arg_list["learning_rate"]
        self.weight_decay = arg_list["weight_decay"]
        self.data_path = arg_list["data_path"]
        self.train_data_path = arg_list["train_data_path"]
        self.eval_data_path = arg_list["eval_data_path"]
        self.bert_model_path = arg_list["bert_model_path"]
        self.tokenizer_max_len = arg_list["tokenizer_max_len"]