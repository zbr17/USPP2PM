# 2022-5-5: fold 5, bert-for-patents, baseline
class config:
    device = torch.device("cuda:0")
    dataset_name = "combined"
    # losses
    loss_name = "pearson" # mse / shift_mse / pearson
    # models
    pretrain_name = None # bert-for-patents / deberta-v3-large
    infer_name = "test"
    # training
    lr = 2e-5
    wd = 0.01
    num_fold = None # 0/1 for training all
    epochs = None
    bs = None
    num_workers = 12
    lr_multi = 10
    sche_step = 5
    sche_decay = 0.5
    # log
    tag = ""