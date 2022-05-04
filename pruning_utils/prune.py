import os
# from copyreg import pickle
from tqdm import tqdm
import torch
import numpy as np
import pickle

def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, 
               store_mask=False, pruner_name='', compression='', dataset='mnist', args=None):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)
    
    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()
    
    if store_mask:
        # pruner_name = args.pruner
        if args.shuffle:
            pruner_name = 'shuffled_' + args.pruner
        if args.pruner in ['grasp', 'snip'] and args.prune_epochs == 100:
            pruner_name = 'iterative_' + pruner_name
        if args.pruner == 'synflow' and args.prune_epochs == 1:
            pruner_name = 'oneshot_' + pruner_name
        
        if args.model is not 'fc':
            file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/{args.init_type}/pre_epoch_{args.pre_epochs}/compression_{int(args.compression*100)}/{pruner_name}'
        else:
            file_path = f'./Reproduced_Results/Masks/{args.dataset}_{args.model}/{args.init_type}/pre_epoch_{args.pre_epochs}/MLP_{args.n_layers}_layers_{args.n_neurons}/compression_{int(args.compression*100)}/{pruner_name}'
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = f'{file_path}/{pruner_name}_{int(compression*100)}.pkl'
        stored_data = {}
        stored_masks = []
        stored_params = []
        for m, p in pruner.masked_parameters:
            stored_masks.append(m.detach().cpu().numpy())
            stored_params.append(p.detach().cpu().numpy())
        stored_data['mask'] = stored_masks
        stored_data['param'] = stored_params
        
        with open(file_name, 'wb') as f:
            pickle.dump(stored_data, f)
