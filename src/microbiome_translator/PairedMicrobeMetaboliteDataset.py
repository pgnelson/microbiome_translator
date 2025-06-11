import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd

class PairedMicrobeMetaboliteDataset(torch.utils.data.Dataset):

    def filter_sparse_columns(self, df, min_frac_nonzero, min_abundance):
        df[df < min_abundance] = 0
        threshold = (min_frac_nonzero) * len(df)
        nonzero_counts = (df != 0).sum(axis=0)
        columns_to_keep = nonzero_counts[nonzero_counts >= threshold].index
        return df[columns_to_keep]
    
    def __init__(self, microbe_df, metabolite_df, 
                 microbe_min_frac_nonzero = 0.2, metabolite_min_frac_nonzero = 0.2, 
                 microbe_min_abundance = 0, metabolite_min_abundance = 0, scale = True, renormalize = True):
        #if it's not a pandas dataframe, convert
        if not isinstance(microbe_df, pd.DataFrame):
            microbe_df = pd.DataFrame(microbe_df.numpy())
        if not isinstance(metabolite_df, pd.DataFrame):
            metabolite_df = pd.DataFrame(metabolite_df.numpy())

        microbe_df = microbe_df.fillna(0)
        metabolite_df = metabolite_df.fillna(0)

        if "O" in list(microbe_df.dtypes):
            print("Please ensure all columns in the microbe dataframe are numeric")
            sys.exit()
        if "O" in list(metabolite_df.dtypes):
            print("Please ensure all columns in the metabolite dataframe are numeric")
            sys.exit()

        self.samples = list(set(microbe_df.index).intersection(set(metabolite_df.index)))
        if len(self.samples) == 0:
            print("No common samples found, please ensure samples are in rows of both dataframes")
            sys.exit()

        microbe_df = self.filter_sparse_columns(microbe_df, microbe_min_frac_nonzero, microbe_min_abundance)
        metabolite_df = self.filter_sparse_columns(metabolite_df, metabolite_min_frac_nonzero, metabolite_min_abundance)

        self.microbe_labels = microbe_df.columns
        self.metabolite_labels = metabolite_df.columns

        microbe_df = microbe_df.loc[self.samples]
        metabolite_df = metabolite_df.loc[self.samples]

        if renormalize:
            microbe_df.div(microbe_df.sum(axis=1), axis=0)
            metabolite_df.div(metabolite_df.sum(axis=1), axis=0)

        if scale:
            scaler = StandardScaler()
            self.microbes = torch.tensor(scaler.fit_transform(microbe_df), dtype=torch.float64)
            self.metabolites = torch.tensor(scaler.fit_transform(metabolite_df), dtype=torch.float64)
        else:
            self.microbes = torch.tensor(microbe_df.values, dtype=torch.float64)
            self.metabolites = torch.tensor(metabolite_df.values, dtype=torch.float64)
    
    def __len__(self):
        return len(self.microbes)
    
    def __getitem__(self, idx):
        return self.microbes[idx], self.metabolites[idx]
