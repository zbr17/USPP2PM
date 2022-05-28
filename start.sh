python train.py --num_fold 5 \
--dataset_name combined \
--pretrain_name deberta-v3-base \
--loss_name mse \
--bs 48 \
--epochs 10 \
--lr 0.00002 \
--wd 1 \
--nproc_per_node 1

python train.py --num_fold 5 \
--dataset_name combined \
--pretrain_name deberta-v3-base \
--loss_name mse \
--bs 48 \
--epochs 10 \
--lr 0.00002 \
--wd 0.1 \
--nproc_per_node 1

python train.py --num_fold 5 \
--dataset_name combined \
--pretrain_name deberta-v3-base \
--loss_name mse \
--bs 48 \
--epochs 10 \
--lr 0.00002 \
--wd 0.01 \
--nproc_per_node 1

python train.py --num_fold 5 \
--dataset_name combined \
--pretrain_name deberta-v3-base \
--loss_name mse \
--bs 48 \
--epochs 10 \
--lr 0.00002 \
--wd 0.001 \
--nproc_per_node 1

python train.py --num_fold 5 \
--dataset_name combined \
--pretrain_name deberta-v3-base \
--loss_name mse \
--bs 48 \
--epochs 10 \
--lr 0.00002 \
--wd 0.0001 \
--nproc_per_node 1