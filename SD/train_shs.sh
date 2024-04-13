
python shs.py --class_to_forget '0' --train_method full --sparsity 0.9 --lam 0.5 --memory_num 1 --prune_num 10 --lr 1e-5 --epochs 5 --device '0' --project
python shs.py --class_to_forget '0' --train_method xattn --sparsity 0.9 --lam 0.1 --memory_num 1 --prune_num 10 --lr 1e-5 --epochs 5 --device '2' --project
python shs.py --class_to_forget '0' --train_method xattn --sparsity 0.9 --lam 0.5 --memory_num 1 --prune_num 10 --lr 1e-5 --epochs 10 --device '3' --project

