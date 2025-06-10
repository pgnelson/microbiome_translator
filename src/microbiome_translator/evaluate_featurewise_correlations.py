def evaluate_featurewise_correlations(model, device='cpu'):
		"""
		Evaluates Spearman correlation between predicted and actual features
		in the dev set, with Benjamini-Hochberg correction for multiple testing.
		Returns:
			microbe_df (pd.DataFrame): feature-wise correlation stats for microbe predictions
			metabolite_df (pd.DataFrame): feature-wise correlation stats for metabolite predictions
		"""
		# Move model to device and set to eval
		model = model.to(device)
		model.eval()

		# Move dev data to device
		microbes = model.dev_dataset.microbes.to(device)
		metabolites = model.dev_dataset.metabolites.to(device)
		
		# Get predictions
		with torch.no_grad():
			pred_microbes = model.microbe_decoder(model.metabolite_encoder(metabolites))
			pred_metabolites = model.metabolite_decoder(model.microbe_encoder(microbes))

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

		microbe_df = compute_correlation_table(pred_microbes, true_microbes, model.all_data.microbe_labels)
		metabolite_df = compute_correlation_table(pred_metabolites, true_metabolites, model.all_data.metabolite_labels)

		return microbe_df, metabolite_df