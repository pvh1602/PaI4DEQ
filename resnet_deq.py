import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging
import argparse
from datetime import datetime
import wandb
import random

from pruning_utils.layers import *
from pruning_utils.prune import *
from pruning_utils.pruners import *
from pruning_utils.generator import *
from utils import *

os.environ["WANDB_API_KEY"] = '5f62978928422ac0179258d5ccb983f8c7b065cf'
os.environ["WANDB_MODE"] = "online"

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

class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))


class MaskedResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8, pruner=''):
        super().__init__()
        self.conv1 = Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
        self.pruner = pruner
        
    def forward(self, z, x):
        if self.pruner == 'grasp':
            y = self.norm1(F.leaky_relu(self.conv1(z)))
            return self.norm3(F.leaky_relu(z + self.norm2(x + self.conv2(y))))
        else:
            y = self.norm1(F.relu(self.conv1(z)))
            return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res


def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res


import torch.autograd as autograd

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z



# CIFAR10 data loader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

cifar10_train = datasets.CIFAR10("data", train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10("data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(cifar10_train, batch_size = 64, shuffle=True, num_workers=1)
test_loader = DataLoader(cifar10_test, batch_size = 64, shuffle=False, num_workers=1)



import tqdm
# standard training or evaluation loop
def epoch(loader, model, device='cpu', opt=None, lr_scheduler=None):
    total_loss, total_err = 0.,0.
    model.eval() if opt is None else model.train()
    forward_iteration = 0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
                
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        # forward_iteration += model.DEQFixedPoint.forward_res
        forward_iteration = 0.        

    return total_err / len(loader.dataset), total_loss / len(loader.dataset), forward_iteration / len(loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning at Initialization for Monotone DEQ')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist','cifar10','cifar100','shvn'],
                        help='dataset (default: mnist)')
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

    device = args.gpu

    # torch.manual_seed(0)
    chan = 48
    f = MaskedResNetLayer(chan, 64, kernel_size=3, pruner=args.pruner)
    model = nn.Sequential(nn.Conv2d(3,chan, kernel_size=3, bias=True, padding=1),
                        nn.BatchNorm2d(chan),
                        DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5),
                        nn.BatchNorm2d(chan),
                        nn.AvgPool2d(8,8),
                        nn.Flatten(),
                        nn.Linear(chan*4*4,10)).to(device)

    logging_path = f'./logging/Resnet_DEQ/{args.pruner}/sparsity_{args.sparsity}'
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    logging_name = f'{logging_path}/{args.pruner}_{args.sparsity}_{date}'
    logging.basicConfig(filename=logging_name, format='%(asctime)s %(message)s', filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # wandb_logger = None
    if args.is_wandb:
        wandb_exp_name = f'sparsity_{args.sparsity}'
        wandb_group_name = 'Cifar10_ResNet_DEQ'
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


    if args.is_pruning:
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
            
            prune_loop(model, 
                    nn.CrossEntropyLoss(),
                    pruner=pruner,
                    dataloader=train_loader,
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
        # try:
        #     # logger.info('The number of params of model is \t ', count_parameters(model))
        # except:
        #     pass
        logger.info('='*40)


        print(f'Total parameters \t {total_params}')
        print(f'Remaining parameters \t {remaining_params}')
        print(f'Expected remaining parameters \t {total_params*args.sparsity}')
        print('='*40)

        print('The number of params of model is \t ', count_parameters(model))

    else:
        print('='*40)
        print('NORMAL TRAINING')
    print(model)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

    max_epochs = args.post_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*len(train_loader), eta_min=1e-6)
    
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    test_loss_history = []
    train_forward_iters_history = []
    test_forward_iters_history = []

    print('='*40)
    print('Training model')
    for i in range(max_epochs):
        train_err, train_loss, train_forward_iters = epoch(train_loader, model, device, opt, scheduler)
        test_err, test_loss, test_forward_iters = epoch(test_loader, model, device)
        train_acc = 1 - train_err
        test_acc = 1 - test_err

        print(f'Epoch {i} \t training accuracy {round(train_acc, 4)} \t test accuracy {round(test_acc, 4)}')
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        test_acc_history.append(test_acc)
        test_loss_history.append(test_loss)
        train_forward_iters_history.append(train_forward_iters)
        test_forward_iters_history.append(test_forward_iters)

        if args.is_wandb:
            wandb.log({'Training accuracy':train_acc}, step=i)
            wandb.log({'Training loss':train_loss}, step=i)
            wandb.log({'Test accuracy':test_acc}, step=i)
            wandb.log({'Test loss':test_loss}, step=i)
            wandb.log({'Training forward iters':train_forward_iters}, step=i)
            wandb.log({'Test forward iters':test_forward_iters}, step=i)

    
    if logger is not None:
        logger.info('Training accuracy history')
        logger.info(train_acc_history)
        logger.info('='*40)
        logger.info('Training loss history')
        logger.info(train_loss_history)
        logger.info('='*40)
        logger.info('Training forward iterations history')
        logger.info(train_forward_iters_history)
        logger.info('='*40)
        logger.info('Test accuracy history')
        logger.info(test_acc_history)
        logger.info('='*40)
        logger.info('Test loss history')
        logger.info(test_loss_history)
        logger.info('='*40)
        logger.info('Test forward iterations history')
        logger.info(test_forward_iters_history)
        logger.info('='*40)