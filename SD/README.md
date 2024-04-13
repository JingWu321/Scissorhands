# Scissorhands for Stable Diffusion
This is the official repository for Scissorhands for diffusion models. The code structure of this project is adapted from the [SalUn](https://github.com/OPTML-Group/Unlearn-Saliency/tree/master/SD) and [SA](https://github.com/clear-nus/selective-amnesia/tree/main/sd) codebase.

# Requirements
Install the requirements using a `conda` environment:
```
conda env create -f environment.yml
```

## Download SD v1.4 Checkpoint
We will use the SD v1.4 checkpoint (with EMA). You can either download it from the official HuggingFace [link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and move it to the root directory of this project, or alternatively
```
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
```

# Forgetting Training with Scissorhands
We consider forgetting the nudity in the following steps.

1. Generate Samples for training.

```
python eval_scripts/generate-images_nude.py --prompts_path 'prompts/nsfw.csv' --save_path 'evaluation_folder/NSFW&NotNSFW/' --model_name 'SD' --device 'cuda:0' --num_samples $N
```

2. Forgetting Training with Scissorhands.

```
python shs.py --train_method full --sparsity 0.99 --lam 0.1 --memory_num 1 --prune_num 1 --lr 5e-5 --epochs 5 --batch_size 16 --device '0'
```


# Evaluation

## NudeNet

1. Generate the images from the I2P prompts.
   
```
python eval_scripts/generate-images_i2p.py --prompts_path 'prompts/unsafe-prompts4703.csv' --save_path 'evaluation_folder/i2p/' --model_name checkpoint --device 'cuda:2' --num_samples 1
```

2. Run the NudeNet evaluation.

```
python nudenet_evaluator.py path/to/outdir
```

The statistics of the relevant nudity concepts will be printed to screen.
