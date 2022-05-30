cd ..

for pruner in rand mag snip grasp synflow
    do 
    for sparsity in 0.1 0.5 0.7 0.9 0.95 0.97 0.99 0.999
        do
        for outchannels in 64 128
            do
                CUDA_VISIBLE_DEVICES=0 python run_exps.py --model SingleConvNet --pruner $pruner --sparsity $sparsity --out_channels $outchannels --dataset cifar10 --is_wandb --is_pruning
            done
        done
    done
