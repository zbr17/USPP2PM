python train.py --num_fold 5 \
--dataset_name combined \
--pretrain_name deberta-v3-base \
--loss_name mse \
--bs 48 \
--epochs 10 \
--nproc_per_node 1