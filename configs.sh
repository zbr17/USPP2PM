

##### SCORED!!!! #############################

# 20220505: fold 1, deberta-v3-large, baseline; SCORE: 0.810
class config:
    device = torch.device("cuda:0")
    dataset_name = "combined"
    # losses
    loss_name = "pearson" # mse / shift_mse / pearson
    # models
    pretrain_name = "deberta-v3-large" # bert-for-patents / deberta-v3-large
    infer_name = "test"
    # training
    lr = 2e-5
    wd = 0.01
    num_fold = 1 # 0/1 for training all
    epochs = None
    bs = None
    num_workers = 12
    lr_multi = 10
    sche_step = 5
    sche_decay = 0.5
    # log
    tag = ""

# 20220505: fold 5, bert-for-patents, baseline; SCORE: 
class config:
    device = torch.device("cuda:0")
    dataset_name = "combined"
    # losses
    loss_name = "pearson" # mse / shift_mse / pearson
    # models
    pretrain_name = "bert-for-patents" # bert-for-patents / deberta-v3-large
    infer_name = "test"
    # training
    lr = 2e-5
    wd = 0.01
    num_fold = 5 # 0/1 for training all
    epochs = None
    bs = None
    num_workers = 12
    lr_multi = 10
    sche_step = 5
    sche_decay = 0.5
    # log
    tag = ""