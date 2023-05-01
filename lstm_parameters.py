args = {
    "kidney": {
        "max_seq_len": 256,
        "emb_dim": 200,
        "hid_dim": 128,
        "bidirectional": True,
        "epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "train_batch_size": 4,
        "test_batch_size": 4,
        "data_path": "PathologyDataset/structured_sampledataset.xlsx",
        "train_data_path": "PathologyDataset/structured_sampledataset_train.xlsx",
        "eval_data_path": "PathologyDataset/structured_sampledataset_eval.xlsx",
        "bert_model_path": "bert-base-uncased",
        "MT_target_cols": ["Anatomical_Position_type", "AJCC_Classification_id",
                           "Histologic_Type_type", "Fuhrman_Nuclear_Grade_type"],
        "MT_renamed_target_cols": ["anatomical_side", "ajcc_classification",
                                   "histologic_classification","fuhrman_type"],
        "MT_other_cols": ["Pateint_ID_text", "TEXT"],
        "tokenizer_max_len": 12}
}


class Args(object):
    def __init__(self, arg_list, type):
        self.max_seq_len = arg_list[type]["max_seq_len"]
        self.emb_dim = arg_list[type]["emb_dim"]
        self.hid_dim = arg_list[type]["hid_dim"]
        self.bidirectional = arg_list[type]["bidirectional"]
        self.epochs = arg_list[type]["epochs"]
        self.learning_rate = arg_list[type]["learning_rate"]
        self.weight_decay = arg_list[type]["weight_decay"]
        self.train_batch_size = arg_list[type]["train_batch_size"]
        self.test_batch_size = arg_list[type]["test_batch_size"]
        self.data_path = arg_list[type]["data_path"]
        self.train_data_path = arg_list[type]["train_data_path"]
        self.eval_data_path = arg_list[type]["eval_data_path"]
        self.bert_model_path = arg_list[type]["bert_model_path"]
        self.MT_target_cols = arg_list[type]["MT_target_cols"]
        self.MT_renamed_target_cols = arg_list[type]["MT_renamed_target_cols"]
        self.MT_other_cols = arg_list[type]["MT_other_cols"]
        self.tokenizer_max_len = arg_list[type]["tokenizer_max_len"]
