cd ../..

for pruner in synflow
    do 
    for sparsity in 0.7 0.9 0.95 0.97 0.99 0.999
    # for sparsity in 0.9 0.95 0.97 0.99 0.999
        do
            CUDA_VISIBLE_DEVICES=1 python run_exps.py --model SingleConvNet --pruner $pruner --sparsity $sparsity --dataset shvn --is_wandb --seed 1 --is_pruning --lr 0.001 --lr_drop_step 25 --prune_epochs 100
        done
    done