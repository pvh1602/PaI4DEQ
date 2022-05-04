# Pruning at Initialization for Monotone DEQ


## Requirements
Compatible with python 3.5+ and known to work with pytorch 1.4, torchvision 0.5, and numpy 1.18. Can install with `pip install -r requirements.txt`.

## Run
`
python run_exps.py --is_pruning --pruner rand --sparsity 0.9 --dataset mnist --model SingleFcNet --sp PR
`

## TODO
- Add mondeq arguments, e.g, out_dim, out_channels, max_iters, etc., 
- Add wandb to log results
- Modify logging name to log with different mondeq arguments 

## Reference 
*MonDEQ codebase is from [the paper](https://arxiv.org/abs/2006.08591) by Ezra Winston and [Zico Kolter](http://zicokolter.com).*

*Pruning codebase is from [the paper](https://arxiv.org/abs/2006.05467) by Tanaka*