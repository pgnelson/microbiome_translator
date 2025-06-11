# microbiome_translator
An encoder to translate between paired metagenomic and metabolomic microbiome data

Microbiome Translator is a Python package for constructing attention-based encoder–decoder models to explore relationships between microbial metagenomic and metabolomic data. It provides modular components for training unsupervised embeddings of microbial and metabolite profiles using multiheaded attention, followed by unsupervised translation between the two domains. The architecture supports autoencoding pretraining, symmetric translation, and evaluation using per-feature Spearman correlations with multiple testing correction. Designed for exploratory multi-omics analyses, this package offers a flexible framework for investigating microbe–metabolite associations without assuming predefined taxonomic or pathway structure.<br/>
The main microbiome_translator function takes two pandas dataframes

# Training
The model was trained on concatinated datasets from Borenstein lab's curated gut microbiome-metabolome project.
https://github.com/borenstein-lab/microbiome-metabolome-curated-data/wiki<br/>
Muller, Efrat, Yadid M. Algavi, and Elhanan Borenstein. "The gut microbiome-metabolome dataset collection: a curated resource for integrative meta-analysis." npj Biofilms and Microbiomes 8.1 (2022): 1-7.

This dataset includes 1,776 samples drawn from 14 different studies. Training the model takes approximately 20 minutes on a local GPU. The model can be trained using the following command:

model = MicrobeMetaboliteTranslator(mgx_df, mbx_df, embed_dim=32, num_heads=4, dev_frac = 0.2, batch_size = 16, microbe_min_frac_nonzero = 0.2, metabolite_min_frac_nonzero = 0.2, microbe_min_abundance = 10**-4, metabolite_min_abundance = 10**-4, scale = True, renormalize = True)<br/>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")<br/>
model.to(device)<br/>
model.pretrain_autoencoders(epochs=2000, lr=1e-3, burn_in=False, device=device)<br/>
model.train_translator(epochs=4000, lr=1e-3, burn_in=False, device = device)<br/>
![image](https://github.com/user-attachments/assets/5d3e45a0-acf2-45ca-a709-7eb347f4019d)<br/>
Note that the first 2000 epochs are training the autoencoders, and the last 4000 are training the translation which is a much harder task.

This yealds a mean spearman r squared between predicted and actual values in the test partition:<br/>
Among microbial species abundances: mean ρ = 0.3101, mean p-value = 0.019
Among annotated metabolites: mean ρ = 0.4359, mean p = 0.0099

Top 5 species best predicted by metabolites:<br/>
| Species |	rho	| pval | qval |
| ---- | ----- | ----- | ----- |
| Odoribacter sp905193145 |	0.167418 | 2.148742e-10 |	3.977892e-10 |
| JAGTTR01 sp018223385|	0.185379	| 1.880935e-12 |	4.015864e-12 |
| UMGS363 sp900768245	| 0.113062	| 1.937690e-05 |	2.623956e-05 |
| Anaeromassilibacillus stercoravium | 0.141075	| 9.292100e-08	| 1.441878e-07 |
| Butyribacter intestini	| 0.075199	| 4.564957e-03 |	5.351703e-03 |

|	Metabolite | rho | pval |	qval |
| ---- | ----- | ----- | ----- |
| 12.13-diHOME |	0.492715 | 8.812994e-88 | 7.709297e-87 |
| 9.10-diHOME	| 0.500094 | 8.772871e-91 | 1.054652e-89 |
| hydrocinnamate | 0.336811 | 4.948382e-39 | 5.859647e-39 |
| mandelate | 0.499328 | 1.812944e-90 | 2.088662e-89 |
| 3-hydroxyoctanoate | 0.503181 | 4.629811e-92 | 6.400714e-91 |

Hydocinnamate is known to correlate with gut microbial diversity (Menni et al. 2020) and mandelate is metabolised by a variety of microbes (Want et al. 2022), so those are promising indicators that model is finding real signal.

# Datasets used:<br/>
Erawijantari et al. Influence of gastrectomy for gastric cancer treatment on faecal microbiome and metabolome profiles. Gut. 2020 Aug;69(8):1404-1415.<br/>
Franzosa, Eric A., et al. "Gut microbiome structure and metabolic activity in inflammatory bowel disease." Nature microbiology 4.2 (2019): 293-305.<br/>
He, Xuan, et al. "Fecal microbiome and metabolome of infants fed bovine MFGM supplemented formula or standard formula with breast-fed infants as reference: a randomized controlled trial." Scientific reports 9.1 (2019): 1-14.<br/>
Jacobs, Jonathan P., et al. "A disease-associated microbial and metabolomics state in relatives of pediatric inflammatory bowel disease patients." Cellular and molecular gastroenterology and hepatology 2.6 (2016): 750-766.<br/>
Kang, Dae-Wook, et al. "Differences in fecal microbial metabolites and microbiota of children with autism spectrum disorders." Anaerobe 49 (2018): 121-131.<br/>
Kim, Minsuk, et al. "Fecal metabolomic signatures in colorectal adenoma patients are associated with gut microbiota and early events of colorectal cancer pathogenesis." MBio 11.1 (2020): e03186-19.<br/>
Kostic, Aleksandar D., et al. "The dynamics of the human infant gut microbiome in development and in progression toward type 1 diabetes." Cell host & microbe 17.2 (2015): 260-273.<br/>
Lloyd-Price, Jason, et al. "Multi-omics of the gut microbial ecosystem in inflammatory bowel diseases." Nature 569.7758 (2019): 655-662.<br/>
Mars, Ruben AT, et al. "Longitudinal multi-omics reveals subset-specific mechanisms underlying irritable bowel syndrome." Cell 182.6 (2020): 1460-1473.<br/>
Poyet, M., et al. "A library of human gut bacterial isolates paired with longitudinal multiomics data enables mechanistic microbiome research." Nature medicine 25.9 (2019): 1442-1452.<br/>
Sinha, Rashmi, et al. "Fecal microbiota, fecal metabolome, and colorectal cancer interrelations." PloS one 11.3 (2016): e0152126.<br/>
Yachida, Shinichi, et al. "Metagenomic and metabolomic analyses reveal distinct stage-specific phenotypes of the gut microbiota in colorectal cancer." Nature medicine 25.6 (2019): 968-976.<br/>
Wandro, Stephen, et al. "The microbiome and metabolome of preterm infant stool are personalized and not driven by health outcomes, including necrotizing enterocolitis and late-onset sepsis." Msphere 3.3 (2018): e00104-18.<br/>
Wang, Xifan, et al. "Aberrant gut microbiota alters host metabolome and impacts renal failure in humans and rodents." Gut 69.12 (2020): 2131-2142.<br/>

# Citations:<br/>
Menni C, Zhu J, Le Roy CI, Mompeo O, Young K, Rebholz CM, Selvin E, North KE, Mohney RP, Bell JT, Boerwinkle E, Spector TD, Mangino M, Yu B, Valdes AM. Serum metabolites reflecting gut microbiome alpha diversity predict type 2 diabetes. Gut Microbes. 2020 Nov 1;11(6):1632-1642. doi: 10.1080/19490976.2020.1778261. Epub 2020 Jun 24. PMID: 32576065; PMCID: PMC7524143.<br/>
Wang, Q., Geng, S., Wang, L., Wen, Z., Sun, X. and Huang, H., 2022. Bacterial mandelic acid degradation pathway and its application in biotechnology. Journal of Applied Microbiology, 133(2), pp.273-286.
