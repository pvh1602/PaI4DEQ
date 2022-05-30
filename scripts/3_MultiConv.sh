cd ..

for pruner in mag
    do 
    for sparsity in 0.1 0.2 0.5 0.6 0.7 0.8 0.99 0.995 0.999
        do
            CUDA_VISIBLE_DEVICES=5 python run_exps.py --model MultiConvNet --pruner $pruner --sparsity $sparsity --dataset cifar10 --is_wandb --is_pruning
        done
    done