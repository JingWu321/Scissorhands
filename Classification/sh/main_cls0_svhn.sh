# # retrain
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/retrain' --unlearn retrain --class_to_replace 0 --unlearn_epochs 50 --unlearn_lr 0.1

# # FT
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/FT' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT --class_to_replace 0 --unlearn_lr 0.01 --unlearn_epochs 10

# # GA
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/GA' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn GA --class_to_replace 0 --unlearn_lr 1e-4 --unlearn_epochs 5

# # IU
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/IU' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn wfisher --class_to_replace 0 --alpha 2

# # BE
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/BE' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_expanding --class_to_replace 0 --unlearn_lr 1e-6 --unlearn_epochs 10

# # BS
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/BS' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn boundary_shrink --class_to_replace 0 --unlearn_lr 1e-6 --unlearn_epochs 10

# # l1_sparse
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/l1Sparse' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn FT_prune --class_to_replace 0 --alpha 1e-4 --unlearn_lr 0.01 --unlearn_epochs 10

# # SalUn
# # python generate_mask.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save './_results/svhn/seed2/SalUn/mask' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --class_to_replace 0 --unlearn_epochs 1
# python -u main_random.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/seed2/SalUn' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --unlearn RL --unlearn_epochs 10 --unlearn_lr 5e-2 --class_to_replace 0 --path './_results/svhn/seed2/SalUn/mask/with_0.5.pt'

# # SHs
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/SVHN' --dataset 'svhn' --save_dir './_results/svhn/cls0/seed2/SHs/lr1e-3_s9_lam0.1_E10P1M1_4' --mask './_results/svhn/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs --class_to_replace 0 --unlearn_epochs 10 --unlearn_lr 1e-3 --sparsity 0.9 --lam 0.1 --project --memory_num 1 --prune_num 1

