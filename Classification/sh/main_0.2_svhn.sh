# # retrain
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/retrain' --unlearn retrain --num_indexes_to_replace 13186 --unlearn_epochs 50 --unlearn_lr 0.1


# # FT
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/FT' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT --num_indexes_to_replace 13186 --unlearn_lr 0.01 --unlearn_epochs 10


# # GA
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/GA' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn GA --num_indexes_to_replace 13186 --unlearn_lr 0.0001 --unlearn_epochs 5


# # IU
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/IU' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn wfisher --num_indexes_to_replace 13186 --alpha 5


# # BE
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/BE' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_expanding --num_indexes_to_replace 13186 --unlearn_lr 0.000001 --unlearn_epochs 10


# # BS
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/BS' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_shrink --num_indexes_to_replace 13186 --unlearn_lr 0.000001 --unlearn_epochs 10


# # l1_sparse
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/l1Sparse' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT_prune --num_indexes_to_replace 13186 --alpha 0.0001 --unlearn_lr 0.01 --unlearn_epochs 10


# # SalUn
# python generate_mask.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save './_results/svhn/seed2/SalUn/mask' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --num_indexes_to_replace 13186 --unlearn_epochs 1
# python -u main_random.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/SalUn' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn RL --unlearn_epochs 10 --unlearn_lr 5e-3 --num_indexes_to_replace 13186 --path './_results/svhn/seed2/SalUn/mask/with_0.5.pt'


# # SHs
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/0.2/seed2/SHs/lr1e-3_s9_lam0.01_E10P1M1' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 13186 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1



