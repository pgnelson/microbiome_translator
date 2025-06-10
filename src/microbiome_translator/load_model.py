def load_model(filepath):
    checkpoint = torch.load(filepath, weights_only = False)

    args = checkpoint['model_args']
    model = MicrobeMetaboliteTranslator(
        args['microbe_data'],
        args['metabolite_data'],
        args['embed_dim'],
        args['num_heads'],
        args['dev_frac'],
        args['batch_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore the loss history
    model.losses = checkpoint.get('losses', [])
    model.train_indices = checkpoint['train_indices']
    model.dev_indices = checkpoint['dev_indices']
    model._build_data_loaders()
    return model

