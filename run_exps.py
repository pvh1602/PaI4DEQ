from ast import arg
import torch
import torch.nn as nn
import copy
import tqdm
import logging
import argparse
from datetime import datetime

import splitting as sp
import train
from pruning_utils.prune import *
from pruning_utils.pruners import *
from pruning_utils.generator import *
from utils import *

import wandb
import random

os.environ["WANDB_API_KEY"] = '5f62978928422ac0179258d5ccb983f8c7b065cf'
os.environ["WANDB_MODE"] = "online"

ablation_mode_mapping = {0: 'Remove_both_A_B',
                        1: 'Remove_B',
                        2: 'Remove_A',
                        3: 'Keep_A_B'} 

def seed_everything(seed=1023):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning at Initialization for Monotone DEQ')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','shvn'],
                        help='dataset (default: mnist)')
    training_args.add_argument('--model', type=str, default='SingleFcNet', choices=['SingleFcNet', 'SingleConvNet', 'MultiConvNet'],
                        help='model architecture (default: SingleFcNet)')
    training_args.add_argument('--train-batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    training_args.add_argument('--test-batch-size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    training_args.add_argument('--pre_epochs', type=int, default=0,
                        help='number of epochs to train before pruning (default: 0)')
    training_args.add_argument('--post_epochs', type=int, default=40,
                        help='number of epochs to train after pruning (default: 10)')
    training_args.add_argument('--seed', type=int, default=2022)
    
    # Pruning Hyperparameters
    pruning_args = parser.add_argument_group('pruning')
    pruning_args.add_argument('--is_pruning', action='store_true', default=False, help='True means pruning')
    pruning_args.add_argument('--is_ablation', action='store_true', default=False, help='True means doing ablation on A, B')
    pruning_args.add_argument('--ablation_mode', type=int, default=0, choices=[0,1,2,3], 
                                help='0: A=0, B=0 | 1: A=full, B=0 | 2: A=0, B=full | 3: A=full, B=full')
    pruning_args.add_argument('--pruner', type=str, default='rand', 
                        choices=['rand','mag','snip','grasp','synflow'],
                        help='prune strategy (default: rand)')
    pruning_args.add_argument('--sparsity', type=float, default=1.0,
                        help='the ratio of total trainable parameters between the pruned network compare to original one (default: 1.0 means unprune)')
    pruning_args.add_argument('--prune_epochs', type=int, default=1,
                        help='number of iterations for scoring (default: 1)')
    pruning_args.add_argument('--pruning-schedule', type=str, default='exponential', choices=['linear','exponential'],
                        help='whether to use a linear or exponential compression schedule (default: exponential)')
    pruning_args.add_argument('--mask-scope', type=str, default='global', choices=['global','local'],
                        help='masking scope (global or layer) (default: global)')
    pruning_args.add_argument('--prune-dataset-ratio', type=int, default=10,
                        help='ratio of prune dataset size and number of classes (default: 10)')
    pruning_args.add_argument('--prune-batch-size', type=int, default=256,
                        help='input batch size for pruning (default: 256)')
    pruning_args.add_argument('--prune-bias', type=bool, default=False,
                        help='whether to prune bias parameters (default: False)')
    pruning_args.add_argument('--prune-batchnorm', type=bool, default=False,
                        help='whether to prune batchnorm layers (default: False)')
    pruning_args.add_argument('--prune-residual', type=bool, default=False,
                        help='whether to prune residual connections (default: False)')
    pruning_args.add_argument('--prune-train-mode', type=bool, default=False,
                        help='whether to prune in train mode (default: False)')
    pruning_args.add_argument('--reinitialize', type=bool, default=False,
                        help='whether to reinitialize weight parameters after pruning (default: False)')
    pruning_args.add_argument('--shuffle', type=bool, default=False,
                        help='whether to shuffle masks after pruning (default: False)')
    pruning_args.add_argument('--invert', type=bool, default=False,
                        help='whether to invert scores during pruning (default: False)')
    pruning_args.add_argument('--store_mask', action='store_true', default=False, help='Storing the mask to file')

    # MonDEQ
    mondeq_args = parser.add_argument_group('mondeq')
    mondeq_args.add_argument('--sp', type=str, default='PR', choices=['PR', 'FB'], help='PR for PeacemanRachford splitting and FB for forwardbackward splitting')
    mondeq_args.add_argument('--out_dim', type=int, default=87, help='Out dim of SingleFcNet for MNIST')
    mondeq_args.add_argument('--out_channels', type=int, default=81, help='Out channel of SingleConvNet')
    mondeq_args.add_argument('--is_augmentation', action='store_true', default=False, help='Use augmentation for CIFAR-10')
    mondeq_args.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    mondeq_args.add_argument('--lr_drop_step', type=int, default=10,
                        help='learning rate step (default: 10)')
    mondeq_args.add_argument('--lr_drop_rate', type=float, default=0.1,
                        help='multiplicative factor of learning rate drop (default: 0.1)')
    mondeq_args.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    mondeq_args.add_argument('--lr_mode', type=str, choices=['1cycle', 'step', 'constant'],
                        default='step', help='Choosing learning rate scheduler')

    parser.add_argument('--is_wandb', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default='0',
                        help='number of GPU device to use (default: 0)')

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.dataset == 'mnist':
        in_dim = 28
        out_dim = 87
        in_channels = 1
        out_channels = 54
        conv_sizes=(16,32,32)

        trainLoader, testLoader = train.mnist_loaders(train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
    
    elif args.dataset == 'cifar10':
        in_dim = 32
        in_channels = 3
        out_channels = args.out_channels
        conv_sizes=(16,32,60)
        if args.is_augmentation:
            conv_sizes=(64,128,128)
            out_channels = 200

        trainLoader, testLoader = train.cifar_loaders(train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size, augment=args.is_augmentation)
    
    elif args.dataset == 'shvn':
        in_dim = 32
        in_channels = 3
        out_channels = args.out_channels
        conv_sizes=(16,32,60)
    
        trainLoader, testLoader = train.svhn_loaders(train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
    
    else:
        raise ValueError(f'Wrong dataset, no dataset named {args.dataset}, expect one of [mnist, cifar10, shvn]')

    if args.sp == 'PR':
        solver = sp.MONPeacemanRachford
    elif args.sp == 'FB':
        solver = sp.MONForwardBackwardSplitting
    else:
        raise ValueError(f'Wrong splitting method, expect [PR, FB]')

    if args.model == 'SingleFcNet':
        if args.pruner == 'grasp' or args.pruner == 'synflow':
            sigmoid_model = train.SingleFcNet(solver, 
                                in_dim=in_dim**2, 
                                out_dim=out_dim, 
                                alpha=1.0,
                                max_iter=300,
                                tol=1e-2,
                                m=1.0,
                                is_pruning=args.is_pruning,
                                pruner=args.pruner)
            sigmoid_model = sigmoid_model.to(args.gpu)
        model = train.SingleFcNet(solver, 
                                in_dim=in_dim**2, 
                                out_dim=out_dim, 
                                alpha=1.0,
                                max_iter=300,
                                tol=1e-2,
                                m=1.0,
                                is_pruning=args.is_pruning)
    elif args.model == 'SingleConvNet':
        if args.pruner == 'grasp' or args.pruner == 'synflow':
            sigmoid_model = train.SingleConvNet(solver,
                                in_dim=in_dim,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                alpha=1.0,
                                max_iter=300,
                                tol=1e-2,
                                m=1.0,
                                is_pruning=args.is_pruning,
                                pruner=args.pruner)
            sigmoid_model = sigmoid_model.to(args.gpu)

        model = train.SingleConvNet(solver,
                                in_dim=in_dim,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                alpha=1.0,
                                max_iter=300,
                                tol=1e-2,
                                m=1.0,
                                is_pruning=args.is_pruning)
    elif args.model == 'MultiConvNet':
        if args.pruner == 'grasp' or args.pruner == 'synflow':
            sigmoid_model = train.MultiConvNet(solver,
                                    in_dim=in_dim,
                                    in_channels=in_channels,
                                    conv_sizes=conv_sizes,
                                    alpha=1.0,
                                    max_iter=300,
                                    tol=1e-2,
                                    m=1.0,
                                    is_pruning=args.is_pruning,
                                    pruner=args.pruner)
            sigmoid_model = sigmoid_model.to(args.gpu)

        model = train.MultiConvNet(solver,
                                in_dim=in_dim,
                                in_channels=in_channels,
                                conv_sizes=conv_sizes,
                                alpha=1.0,
                                max_iter=300,
                                tol=1e-2,
                                m=1.0,
                                is_pruning=args.is_pruning)
    else:
        raise ValueError('Wrong model, expect [SingleFcNet, SingleConvNet, MultiConvNet]')
    
    model = model.to(args.gpu)
    
    if args.model == 'SingleFcNet':
        logging_path = f'./logging/{args.dataset}/{args.model}/out_dim_{args.out_dim}/{args.pruner}/sparsity_{args.sparsity}'
        wandb_group_name = f'{args.dataset}_{args.model}_out_dim_{args.out_dim}'
    elif args.model == 'SingleConvNet':
        logging_path = f'./logging/{args.dataset}/{args.model}/out_channels_{args.out_channels}/{args.pruner}/sparsity_{args.sparsity}'
        wandb_group_name = f'{args.dataset}_{args.model}_out_channels_{args.out_channels}'
    else:
        logging_path = f'./logging/{args.dataset}/{args.model}/conv_sizes_{conv_sizes[0]}_{conv_sizes[1]}_{conv_sizes[2]}/{args.pruner}/sparsity_{args.sparsity}'
        wandb_group_name = f'{args.dataset}_{args.model}_conv_sizes_{conv_sizes[0]}_{conv_sizes[1]}_{conv_sizes[2]}'
    if args.is_ablation:
        wandb_group_name = 'ablation_' + wandb_group_name
        logging_path = logging_path.replace(f'{args.pruner}/sparsity_{args.sparsity}', 'ablation')

    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    logging_name = f'{logging_path}/{args.pruner}_{args.sparsity}_{date}'
    if args.is_ablation:
        logging_name = f'{logging_path}/{ablation_mode_mapping[args.ablation_mode]}'
    logging.basicConfig(filename=logging_name, format='%(asctime)s %(message)s', filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # wandb_logger = None
    if args.is_wandb:
        wandb_exp_name = f'sparsity_{args.sparsity}'

        if args.is_ablation:
            wandb_exp_name = ablation_mode_mapping[args.ablation_mode]
            wandb.init(
            project=f'seed_{args.seed}_PaI4MONDeq',
            entity="pai4deq",
            group=wandb_group_name,
            name=wandb_exp_name,
            config=args
            )
        else:
            wandb.init(
                project=f'seed_{args.seed}_PaI4MONDeq',
                entity="pai4deq",
                group=wandb_group_name,
                name=wandb_exp_name,
                job_type=f'{args.pruner}',
                config=args
            )
    else:
        wandb = None

    if args.is_pruning and not args.is_ablation:
        if args.pruner == 'grasp' or args.pruner == 'synflow':
            masked_parameters_ = masked_parameters(sigmoid_model)
        else:
            masked_parameters_ = masked_parameters(model)

        if args.pruner == 'rand': pruner = Rand(masked_parameters_)
        elif args.pruner == 'mag': pruner = Mag(masked_parameters_)
        elif args.pruner == 'snip': pruner = SNIP(masked_parameters_)
        elif args.pruner == 'grasp': pruner = GraSP(masked_parameters_)
        elif args.pruner == 'synflow': pruner = SynFlow(masked_parameters_) 
        else:
            raise ValueError('Wrong pruner name, expect [rand, mag, snip, grasp, synflow]')        
        # change to the sparsity => TODO: Make it simple by modifing in pruner and prune_loop
        args.sparsity = 1 - args.sparsity
        with torch.autograd.set_detect_anomaly(True):
            if args.pruner == 'grasp' or args.pruner == 'synflow':
                print(sigmoid_model)
                # print(model)

                prune_loop(sigmoid_model, 
                    nn.CrossEntropyLoss(),
                    pruner=pruner,
                    dataloader=trainLoader,
                    device=args.gpu,
                    sparsity=args.sparsity,
                    schedule=args.pruning_schedule,
                    scope=args.mask_scope, 
                    epochs=args.prune_epochs,
                    reinitialize=args.reinitialize, 
                    train_mode=False, 
                    shuffle=args.shuffle, 
                    invert=args.invert
                )
            else:
                prune_loop(model, 
                        nn.CrossEntropyLoss(),
                        pruner=pruner,
                        dataloader=trainLoader,
                        device=args.gpu,
                        sparsity=args.sparsity,
                        schedule=args.pruning_schedule,
                        scope=args.mask_scope, 
                        epochs=args.prune_epochs,
                        reinitialize=args.reinitialize, 
                        train_mode=False, 
                        shuffle=args.shuffle, 
                        invert=args.invert
                        )

        remaining_params, total_params = pruner.stats()
        logger.info(f'Total parameters \t {total_params}')
        logger.info(f'Remaining parameters \t {remaining_params}')
        logger.info(f'Expected remaining parameters \t {total_params*args.sparsity}')
        logger.info('='*40)

        print(f'Total parameters \t {total_params}')
        print(f'Remaining parameters \t {remaining_params}')
        print(f'Expected remaining parameters \t {total_params*args.sparsity}')
        print('='*40)

        print('The number of params is \t ', count_parameters(model))
        exit()

        if args.pruner == 'grasp' or args.pruner == 'synflow':
            # Assign mask from sigmoid model to model
            with torch.no_grad():
                for (n1, p1), (n2, p2) in zip(model.named_parameters(), sigmoid_model.named_parameters()):
                    p1.copy_(p2.data)
                for (n1, m1), (n2, m2) in zip(model.named_buffers(), sigmoid_model.named_buffers()):
                    m1.copy_(m2.data)
            del sigmoid_model

        for n, m in model.named_buffers():
            sparsity = m.sum() / m.numel()
            print(f"{n} \t {sparsity}")
    # Train model 
        # exit()

    elif args.is_ablation and args.is_pruning:
        print('='*40)
        print('RUN ABLATION')
        # help='0: A=0, B=0 | 1: A=full, B=0 | 2: A=0, B=full | 3: A=full, B=full')
        with torch.no_grad():
            if args.ablation_mode == 0: 
                print('Remove both A and B')
                for n, m in model.named_buffers():
                    if 'A' in n or 'B' in n:
                        m.copy_(torch.zeros_like(m))
            elif args.ablation_mode == 1:
                print('Remove B')
                for n, m in model.named_buffers():
                    if 'B' in n:
                        m.copy_(torch.zeros_like(m))
            elif args.ablation_mode == 2:
                print('Remove A')
                for n, m in model.named_buffers():
                    if 'A_n' in n:
                        print(n)
                        m.copy_(torch.zeros_like(m))
            elif args.ablation_mode == 3:
                print('Keep both A and B')
                for n, m in model.named_buffers():
                    if 'A' in n or 'B' in n:
                        m.copy_(torch.ones_like(m))
            else:
                raise ValueError('Wrong ablation mode')          

        remaining_params, total_params = stats(model)
        logger.info(f'Total parameters \t {total_params}')
        logger.info(f'Remaining parameters \t {remaining_params}')
        # logger.info(f'Expected remaining parameters \t {total_params*args.sparsity}')
        logger.info('='*40)

    else:
        print('='*40)
        print('NORMAL TRAINING')
    print(model)
    train.train(trainLoader, 
                testLoader,
                model, 
                max_lr=args.lr,
                lr_mode=args.lr_mode,
                step=args.lr_drop_step,
                change_mo=False,
                epochs=args.post_epochs,
                print_freq=100,
                tune_alpha=True,
                logger=logger,
                wandb=wandb,
                )