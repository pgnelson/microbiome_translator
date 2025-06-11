import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from scipy.stats import spearmanr
import random
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import sys
from microbiome_translator import PairedMicrobeMetaboliteDataset, MultiHeadAttentionEncoder, FeedForwardDecoder
torch.set_default_dtype(torch.float64)


class microbiome_translator(nn.Module):
    def __init__(self, microbe_data, metabolite_data, embed_dim=32, num_heads=4, dev_frac = 0.2, batch_size = 2,
                  microbe_min_frac_nonzero = 0.2, metabolite_min_frac_nonzero = 0.2, 
                    microbe_min_abundance = 0, metabolite_min_abundance = 0, 
                    scale = True, renormalize = True):
        super().__init__()
        print("Preprocessesing Data")
        self.all_data = PairedMicrobeMetaboliteDataset(microbe_data, metabolite_data,
                                                microbe_min_frac_nonzero, metabolite_min_frac_nonzero, 
                                                microbe_min_abundance, metabolite_min_abundance,
                                                scale, renormalize)
        
        self.dev_frac = dev_frac
        self.batch_size = batch_size
        n_samples = len(self.all_data.samples)
        randomized_samples = list(range(0, n_samples))
        random.shuffle(randomized_samples)

        train_indices = randomized_samples[0:int(len(randomized_samples)*dev_frac)]
        dev_indices = randomized_samples[int(len(randomized_samples)*dev_frac):len(randomized_samples)]
        self.train_indices = train_indices
        self.dev_indices = dev_indices
        self._build_data_loaders()

        microbe_dim = self.all_data.microbes.shape[1]
        metabolite_dim = self.all_data.metabolites.shape[1]
        self.microbe_encoder = MultiHeadAttentionEncoder(microbe_dim, embed_dim, num_heads)
        self.metabolite_encoder = MultiHeadAttentionEncoder(metabolite_dim, embed_dim, num_heads)
        self.microbe_decoder = FeedForwardDecoder(embed_dim, microbe_dim)
        self.metabolite_decoder = FeedForwardDecoder(embed_dim, metabolite_dim)
        self.translator = FeedForwardDecoder(embed_dim, metabolite_dim)
        self.microbe_dim = microbe_dim
        self.metabolite_dim = metabolite_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.apply(self.initialize_weights)
        self.losses = []



    def _build_data_loaders(self):
        self.train_dataset = PairedMicrobeMetaboliteDataset(self.all_data.microbes[self.train_indices,:], self.all_data.metabolites[self.train_indices,:],
                                                            0,0,0,0,False,False)
        self.dev_dataset = PairedMicrobeMetaboliteDataset(self.all_data.microbes[self.dev_indices,:], self.all_data.metabolites[self.dev_indices,:],
                                                            0,0,0,0,False,False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False)

    import torch.nn as nn

    def initialize_weights(self, module):
        if isinstance(module, nn.Embedding):
            init.xavier_uniform_(module.weight)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

        elif isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0.)

        elif isinstance(module, nn.MultiheadAttention):
            init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                init.constant_(module.in_proj_bias, 0.)
            init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                init.constant_(module.out_proj.bias, 0.)

    def forward_autoencode_microbe(self, x):
        z = self.microbe_encoder(x)
        return self.microbe_decoder(z)

    def forward_autoencode_metabolite(self, x):
        z = self.metabolite_encoder(x)
        return self.metabolite_decoder(z)

    def forward_translate_microbes(self, microbe_input):
        z = self.microbe_encoder(microbe_input)
        return self.translator(z)
    
    def forward_translate_metabolites(self, microbe_input):
        z = self.microbe_encoder(microbe_input)
        return self.translator(z)

    def shuffle_rows(self, x: torch.Tensor) -> torch.Tensor:
        shuffled = torch.empty_like(x)
        for i in range(x.size(0)):
            perm = torch.randperm(x.size(1))
            shuffled[i] = x[i, perm]
        return shuffled

    def pretrain_autoencoders(self, epochs=10, lr=1e-3, burn_in = False, device = "cpu"):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            sum_loss = 0.0
            all_mic_preds = []
            all_mic_targets = []
            all_met_preds = []
            all_met_targets = []
            for mic_batch, met_batch in self.train_loader:
                opt.zero_grad()
                #mic_df = torch.transpose(mic_batch, 0, 1).to(device)
                #met_df = torch.transpose(met_batch, 0, 1).to(device)
                mic_df = mic_batch.to(device)
                met_df = met_batch.to(device)
                if burn_in:
                    mic_df = self.shuffle_rows(mic_df)
                    met_df = self.shuffle_rows(met_df)
                mic_pred = self.forward_autoencode_microbe(mic_df)
                met_pred = self.forward_autoencode_metabolite(met_df)
                loss_mic = loss_fn(mic_pred, mic_df)
                loss_met = loss_fn(met_pred, met_df)
                loss = loss_mic + loss_met

                all_mic_preds.append(mic_pred.cpu())     # detach and move to CPU
                all_mic_targets.append(mic_df.cpu())
                all_met_preds.append(met_pred.cpu())     
                all_met_targets.append(met_df.cpu())
                # Combine losses
                loss.backward()
                sum_loss += loss
                opt.step()

            self.losses.append(sum_loss.detach().cpu().numpy())
            all_mic_preds = torch.cat(all_mic_preds, dim=0)
            all_mic_targets = torch.cat(all_mic_targets, dim=0)
            all_met_preds = torch.cat(all_met_preds, dim=0)
            all_met_targets = torch.cat(all_met_targets, dim=0)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                self.report_performance(
                    label="Pretrain Encoder Train",
                    epoch=epoch,
                    pred_microbe=all_mic_preds,
                    true_microbe=all_mic_targets,
                    pred_metabolite=all_met_preds,
                    true_metabolite=all_met_targets
                )
                self.report_performance(
                    label="Pretrain Encoder Dev",
                    epoch=epoch,
                    pred_microbe=self.forward_autoencode_microbe(self.dev_dataset.microbes.to(device)),
                    true_microbe=self.dev_dataset.microbes.to(device),
                    pred_metabolite=self.forward_autoencode_metabolite(self.dev_dataset.metabolites.to(device)),
                    true_metabolite=self.dev_dataset.metabolites.to(device)
                )

    def train_translator(self, epochs, lr=1e-3, burn_in = False, device = "cpu"):
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(epochs):
            sum_loss = 0.0
            all_mic_preds = []
            all_mic_targets = []
            all_met_preds = []
            all_met_targets = []
            for mic_batch, met_batch in self.train_loader:
                opt.zero_grad()
                #mic_df = torch.transpose(mic_batch, 0, 1).to(device)
                #met_df = torch.transpose(met_batch, 0, 1).to(device)
                mic_df = mic_batch.to(device)
                met_df = met_batch.to(device)
                if burn_in:
                    mic_df = self.shuffle_rows(mic_df)
                    met_df = self.shuffle_rows(met_df)
                # Microbes -> Metabolites
                met_pred = self.metabolite_decoder(self.microbe_encoder(mic_df))
                loss_forward = loss_fn(met_pred, met_df)
                # Metabolites -> Microbes
                mic_pred = self.microbe_decoder(self.metabolite_encoder(met_df))
                loss_reverse = loss_fn(mic_pred, mic_df)

                all_mic_preds.append(mic_pred.cpu())     # detach and move to CPU
                all_mic_targets.append(mic_df.cpu())
                all_met_preds.append(met_pred.cpu())    
                all_met_targets.append(met_df.cpu())
                # Combine losses
                total_loss = loss_forward + loss_reverse
                total_loss.backward()
                sum_loss += total_loss
                opt.step()

            self.losses.append(sum_loss.detach().cpu().numpy())
            all_mic_preds = torch.cat(all_mic_preds, dim=0)
            all_mic_targets = torch.cat(all_mic_targets, dim=0)
            all_met_preds = torch.cat(all_met_preds, dim=0)
            all_met_targets = torch.cat(all_met_targets, dim=0)
            if (epoch + 1) % 100 == 0 or epoch == 0:
                self.report_performance(
                    label="Translation Training Train",
                    epoch=epoch,
                    pred_microbe=all_mic_preds,
                    true_microbe=all_mic_targets,
                    pred_metabolite=all_met_preds,
                    true_metabolite=all_met_targets
                )
                self.report_performance(
                    label="Translation Training Dev",
                    epoch=epoch,
                    pred_microbe=self.microbe_decoder(self.metabolite_encoder(self.dev_dataset.metabolites.to(device))),
                    true_microbe=self.dev_dataset.microbes.to(device),
                    pred_metabolite=self.metabolite_decoder(self.microbe_encoder(self.dev_dataset.microbes.to(device))),
                    true_metabolite=self.dev_dataset.metabolites.to(device)
                )
    def evaluate_featurewise_correlations(self, device='cpu'):
            """
            Evaluates Spearman correlation between predicted and actual features
            in the dev set, with Benjamini-Hochberg correction for multiple testing.
            Returns:
                microbe_df (pd.DataFrame): feature-wise correlation stats for microbe predictions
                metabolite_df (pd.DataFrame): feature-wise correlation stats for metabolite predictions
            """
            #make sure everything on the same device
            self = self.to(device)
            self.eval()
            microbes = self.dev_dataset.microbes.to(device)
            metabolites = self.dev_dataset.metabolites.to(device)
            
            # Get predictions
            with torch.no_grad():
                pred_microbes = self.microbe_decoder(self.metabolite_encoder(metabolites))
                pred_metabolites = self.metabolite_decoder(self.microbe_encoder(microbes))

            # Move back to CPU for evaluation
            pred_microbes = pred_microbes.cpu().numpy()
            true_microbes = microbes.cpu().numpy()
            pred_metabolites = pred_metabolites.cpu().numpy()
            true_metabolites = metabolites.cpu().numpy()

            def compute_correlation_table(pred, true, labels):
                rhos, pvals = [], []
                for i in range(pred.shape[1]):
                    if np.std(true[:, i]) == 0:
                        rho, p = np.nan, np.nan
                    else:
                        rho, p = spearmanr(pred[:, i], true[:, i])
                    rhos.append(rho)
                    pvals.append(p)

                # FDR correction
                _, qvals, _, _ = multipletests(pvals, method='fdr_bh')

                return pd.DataFrame({
                    'feature': labels,
                    'rho': rhos,
                    'pval': pvals,
                    'qval': qvals
                })

            microbe_df = compute_correlation_table(pred_microbes, true_microbes, self.all_data.microbe_labels)
            metabolite_df = compute_correlation_table(pred_metabolites, true_metabolites, self.all_data.metabolite_labels)

            return microbe_df, metabolite_df

    def save_model(self, filepath):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_args': {
                'microbe_data': self.all_data.microbes,
                'metabolite_data': self.all_data.metabolites,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'batch_size': self.batch_size,
                'dev_frac': self.dev_frac
            },
            'losses': self.losses,  # save the loss history
            'train_indices': self.train_indices,
            'dev_indices': self.dev_indices,
            'microbe_labels': self.all_data.microbe_labels,
            'metabolite_labels': self.all_data.metabolite_labels,
        }
        torch.save(checkpoint, filepath)
