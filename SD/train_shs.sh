python shs.py --class_to_forget '0' --train_method full --sparsity 0.995 --lam 0.1 --memory_num 1 --prune_num 10 --lr 1e-5 --epochs 5 --batch_size 16 --device '0'

python shs_cls.py --class_to_forget '0' --train_method full --sparsity 0.999 --lam 0.1 --memory_num 1 --prune_num 10 --lr 2e-5 --epochs 3 --device '0'
