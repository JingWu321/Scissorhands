# Scissorhands for Classification
This is the official repository for Scissorhands for Clasification. The code structure of this project is adapted from the [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency) codebase.



# Unlearning
### Scissorhands
```
python -u main_forget.py --data ${data_path} --save_dir ${save_dir} --mask ${mask_file} --print_freq=10 --unlearn SHs --num_indexes_to_replace 4500 --unlearn_epochs 10 --unlearn_lr 5e-3 --sparsity 0.97 --lam 0.05 --project --memory_num 1 --prune_num 1
```


