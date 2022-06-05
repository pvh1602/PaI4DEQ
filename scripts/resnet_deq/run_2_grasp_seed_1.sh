cd ../..


for pruner in grasp
do
    for sparsity in 0.1 0.5 0.7 0.9 0.95 0.97 0.99 0.999
    do
        CUDA_VISIBLE_DEVICES=2 python resnet_deq.py --is_pruning --sparsity $sparsity --pruner $pruner --is_wandb --seed 1
    done

done