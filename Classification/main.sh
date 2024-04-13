# 0.1 svhn
python -u main_forget.py --seed=2 --gpu 1 --data '/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/0.1/seed2/SHs/lr1e-3_s9_lam0.01_E10P1M1' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1


# 0.1 cifar10
python -u main_forget.py --seed=3 --gpu 0 --data '/datasets/CIFAR10' --save_dir './_results/cifar10/0.1/seed3/SHs/lr5e-3_s97_lam0.05_E10P1M1' --mask './_results/cifar10/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 4500 --unlearn_epochs 10 --unlearn_lr 5e-3 --sparsity 0.97 --lam 0.05 --project --memory_num 1 --prune_num 1


# 0.1 cifar100
python -u main_forget.py --seed=2 --gpu 0 --data '/datasets/CIFAR100' --dataset 'cifar100' --save_dir './_results/cifar100/seed2/SHs/lr5e-3_s99_lam1_E10P1M1' --mask './_results/cifar100/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 4500 --unlearn_epochs 10 --unlearn_lr 5e-3 --sparsity 0.99 --lam 1. --project --memory_num 1 --prune_num 1


# celebahq
python -u main_forget.py --seed=2 --gpu 1 --data '/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/0.1/seed2/SHs/lr2e-4_s995_lam0.3_E5P1M1' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs  --unlearn_epochs 5 --unlearn_lr 2e-4 --sparsity 0.995 --lam 0.3 --project --memory_num 1 --prune_num 1 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307

