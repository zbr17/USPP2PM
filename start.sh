python train.py --num_fold 10 \
--dataset_name split \
--pretrain_name deberta-v3-large \
--loss_name mse \
--bs 20 \
--epochs 10 \
--nproc_per_node 2