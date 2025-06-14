import torch
from pathlib import Path
from microbiome_translator import microbiome_translator

def load_model(file_or_path):
    if isinstance(file_or_path, (str, Path)):
        checkpoint = torch.load(file_or_path, map_location='cpu',weights_only=False)
    else:
        checkpoint = torch.load(file_or_path, map_location='cpu',weights_only=False)
    args = checkpoint['model_args']
    model = microbiome_translator(
        args['microbe_data'],
        args['metabolite_data'],
        embed_dim = args['embed_dim'],
        num_heads = args['num_heads'],
        dev_frac = args['dev_frac'],
        batch_size = args['batch_size'],
        microbe_min_frac_nonzero = 0,
        metabolite_min_frac_nonzero = 0,
        microbe_min_abundance = 0,
        metabolite_min_abundance = 0,
        scale = False,
        renormalize = False
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore the loss history
    model.losses = checkpoint.get('losses', [])
    model.train_indices = checkpoint['train_indices']
    model.dev_indices = checkpoint['dev_indices']
    model.all_data.microbe_labels = checkpoint['microbe_labels']
    model.all_data.metabolite_labels = checkpoint['metabolite_labels']
    model._build_data_loaders()
    return model
