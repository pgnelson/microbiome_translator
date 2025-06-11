import torch
import importlib.resources as pkg_resources
import microbiome_translator
import microbiome_translator.resources
from microbiome_translator import load_model, trained_model
import pandas as pd

def example_load():
    # Get the path to the bundled model
    with pkg_resources.path(microbiome_translator.resources, "trained_microbiome_translator_borenstein.pth") as model_path:
        model = load_model(model_path)
        return model
trained_model = example_load()
sp_cor_df, met_cor_df = trained_model.evaluate_featurewise_correlations("cpu")

sp_cor_df['Species'] = sp_cor_df.feature.str.split("s__").str[-1]
sp_cor_df.sort_values('rho', ascending=True)
print(sp_cor_df.head())

met_cor_df.sort_values('rho', ascending=True)
print(met_cor_df.head())
