import torch
def load_model(filepath):
    checkpoint = torch.load(filepath, weights_only = False)
    args = checkpoint['model_args']
    model = MicrobeMetaboliteTranslator(
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
    model._build_data_loaders()
    return model
