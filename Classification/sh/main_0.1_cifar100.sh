
# # raw
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/raw' --unlearn raw --num_indexes_to_replace 4500 --unlearn_epochs 185 --unlearn_lr 0.1


# # retrain
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/retrain' --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1


# # FT
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/FT' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10


# # GA
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/GA' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5


# # IU
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/IU' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --unlearn wfisher --num_indexes_to_replace 4500 --alpha 20


# # BE
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/BE' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --unlearn boundary_expanding --num_indexes_to_replace 4500 --unlearn_lr 0.000001 --unlearn_epochs 10


# # BS
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/BS' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --unlearn boundary_shrink --num_indexes_to_replace 4500 --unlearn_lr 0.000001 --unlearn_epochs 10


# # l1_sparse
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/l1Sparse' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --unlearn FT_prune --num_indexes_to_replace 4500 --alpha 0.0001 --unlearn_lr 0.01 --unlearn_epochs 10


# # SalUn
# python generate_mask.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save './_results/cifar100/seed2/SalUn/mask' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --num_indexes_to_replace 4500 --unlearn_epochs 1
# python -u main_random.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/SalUn' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --unlearn RL --unlearn_epochs 10 --unlearn_lr 5e-3 --num_indexes_to_replace 4500 --path './_results/cifar100/seed2/SalUn/mask/with_0.5.pt'


# # SHs
# python -u main_forget.py --seed=2 --gpu 0 --data '/data/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/SHs/lr5e-3_s99_lam1_E10P1M1' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 4500 --unlearn_epochs 10 --unlearn_lr 5e-3 --sparsity 0.99 --lam 1. --project --memory_num 1 --prune_num 1

