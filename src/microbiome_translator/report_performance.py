from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

def report_performance(label, epoch, pred_microbe, true_microbe, pred_metabolite, true_metabolite):
    def get_spearman(pred, true):
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        rho_list, pval_list = [], []

        for i in range(true.shape[0]):
            rho, pval = spearmanr(pred[i], true[i])
            if not np.isnan(rho):
                rho_list.append(rho)
                pval_list.append(pval)

        mean_rho = np.mean(rho_list)
        mean_p = np.mean(pval_list)
        return mean_rho, mean_p

    rho_microbe, p_microbe = get_spearman(pred_microbe, true_microbe)
    rho_metabolite, p_metabolite = get_spearman(pred_metabolite, true_metabolite)

    print(
        f"[{label} | Epoch {epoch+1}] "
        f"Microbes: ρ={rho_microbe:.4f}, p={p_microbe:.2g} | "
        f"Metabolites: ρ={rho_metabolite:.4f}, p={p_metabolite:.2g}"
    )
