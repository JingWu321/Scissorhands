
# retrain
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/retrain' --unlearn retrain --num_indexes_to_replace 6593 --unlearn_epochs 50 --unlearn_lr 0.1


# FT
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/FT' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT --num_indexes_to_replace 6593 --unlearn_lr 0.01 --unlearn_epochs 10


# GA
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/GA' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn GA --num_indexes_to_replace 6593 --unlearn_lr 0.0001 --unlearn_epochs 5


# IU
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/IU' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn wfisher --num_indexes_to_replace 6593 --alpha 15


# BE
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/BE' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_expanding --num_indexes_to_replace 6593 --unlearn_lr 0.000001 --unlearn_epochs 10


# BS
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/BS' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_shrink --num_indexes_to_replace 6593 --unlearn_lr 0.000001 --unlearn_epochs 10


# l1_sparse
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/l1Sparse' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT_prune --num_indexes_to_replace 6593 --alpha 0.0001 --unlearn_lr 0.01 --unlearn_epochs 10


# # SalUn
# python generate_mask.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save './_results/svhn/seed2/SalUn/mask' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --num_indexes_to_replace 6593 --unlearn_epochs 1
python -u main_random.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/SalUn' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn RL --unlearn_epochs 10 --unlearn_lr 5e-3 --num_indexes_to_replace 6593 --path './_results/svhn/seed2/SalUn/mask/with_0.5.pt'


# SHs
python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/SHs' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1





# retrain
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/retrain' --unlearn retrain --num_indexes_to_replace 6593 --unlearn_epochs 160 --unlearn_lr 0.1


# FT
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/FT' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT --num_indexes_to_replace 6593 --unlearn_lr 0.01 --unlearn_epochs 10


# GA
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/GA' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn GA --num_indexes_to_replace 6593 --unlearn_lr 0.0001 --unlearn_epochs 5


# IU
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/IU' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn wfisher --num_indexes_to_replace 6593 --alpha 10


# BE
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/BE' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_expanding --num_indexes_to_replace 6593 --unlearn_lr 0.000001 --unlearn_epochs 10


# BS
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/BS' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_shrink --num_indexes_to_replace 6593 --unlearn_lr 0.000001 --unlearn_epochs 10


# l1_sparse
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/l1Sparse' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT_prune --num_indexes_to_replace 6593 --alpha 0.0001 --unlearn_lr 0.01 --unlearn_epochs 10


# # SalUn
# python generate_mask.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save './_results/svhn/seed3/SalUn/mask' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --num_indexes_to_replace 6593 --unlearn_epochs 1
python -u main_random.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/SalUn' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn RL --unlearn_epochs 10 --unlearn_lr 5e-3 --num_indexes_to_replace 6593 --path './_results/svhn/seed3/SalUn/mask/with_0.5.pt'


# SHs
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed3/SHs' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 5e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1



# retrain
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/retrain' --unlearn retrain --num_indexes_to_replace 6593 --unlearn_epochs 160 --unlearn_lr 0.1


# FT
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/FT' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT --num_indexes_to_replace 6593 --unlearn_lr 0.01 --unlearn_epochs 10


# GA
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/GA' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn GA --num_indexes_to_replace 6593 --unlearn_lr 0.0001 --unlearn_epochs 5


# IU
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/IU' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn wfisher --num_indexes_to_replace 6593 --alpha 10


# BE
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/BE' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_expanding --num_indexes_to_replace 6593 --unlearn_lr 0.000001 --unlearn_epochs 10


# BS
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/BS' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_shrink --num_indexes_to_replace 6593 --unlearn_lr 0.000001 --unlearn_epochs 10


# l1_sparse
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/l1Sparse' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT_prune --num_indexes_to_replace 6593 --alpha 0.0001 --unlearn_lr 0.01 --unlearn_epochs 10


# # SalUn
# python generate_mask.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save './_results/svhn/seed4/SalUn/mask' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --num_indexes_to_replace 6593 --unlearn_epochs 1
python -u main_random.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/SalUn' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn RL --unlearn_epochs 10 --unlearn_lr 5e-3 --num_indexes_to_replace 6593 --path './_results/svhn/seed4/SalUn/mask/with_0.5.pt'


# SHs
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed4/SHs' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 5e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1




# # SHs
python -u main_forget.py --seed=1 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/0.1/seed1/SHs/lr1e-3_s9_lam0.01_E10P1M1' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1
python -u main_forget.py --seed=3 --gpu 0 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/0.1/seed3/SHs/lr1e-3_s9_lam0.01_E10P1M1' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1

python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/0.1/seed2/SHs/lr1e-3_s9_lam0.01_E10P1M1' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1
python -u main_forget.py --seed=4 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/0.1/seed4/SHs/lr1e-3_s9_lam0.01_E10P1M1' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --num_indexes_to_replace 6593 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.01 --project --memory_num 1 --prune_num 1

