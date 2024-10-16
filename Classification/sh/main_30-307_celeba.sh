# # raw
# python -u main_forget.py --seed=2 --gpu 0 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/raw' --unlearn raw --unlearn_epochs 10 --unlearn_lr 1e-3 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307

# # retrain
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/retrain' --unlearn retrain  --unlearn_epochs 10 --unlearn_lr 1e-3 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # FT
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/FT' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --unlearn FT  --unlearn_lr 1e-4 --unlearn_epochs 5 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # GA
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/GA' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --unlearn GA  --unlearn_lr 1e-4 --unlearn_epochs 3 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # IU
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/IU' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --unlearn wfisher  --alpha 5 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # BE
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/BE' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --unlearn boundary_expanding  --unlearn_lr 1e-5 --unlearn_epochs 5 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # BS
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/BS' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --unlearn boundary_shrink  --unlearn_lr 1e-5 --unlearn_epochs 5 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # l1_sparse
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/l1Sparse' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --unlearn FT_prune  --alpha 1e-4 --unlearn_lr 1e-3 --unlearn_epochs 5 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # SalUn
# python generate_mask.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save './_results/celeba/seed2/SalUn/mask' --mask './_results/celeba/raw/rawcheckpoint.pth.tar'  --unlearn_epochs 1 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307
# python -u main_random.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/SalUn' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --unlearn RL --unlearn_epochs 5 --unlearn_lr 5e-3  --path './_results/celeba/seed2/SalUn/mask/with_0.2.pt' --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307


# # SHs
# python -u main_forget.py --seed=2 --gpu 1 --data '/home/jing/datasets/CelebAMaskHQ/CelebA_HQ_facial_identity_dataset' --dataset 'celeba' --save_dir './_results/celeba/seed2/SHs/lr2e-4_s995_lam0.3_E5P1M1' --mask './_results/celeba/raw/rawcheckpoint.pth.tar' --print_freq=10 --unlearn SHs  --unlearn_epochs 5 --unlearn_lr 2e-4 --sparsity 0.995 --lam 0.3 --project --memory_num 1 --prune_num 1 --arch "resnet34" --input_size 224 --batch_size 8 --num_classes 307
