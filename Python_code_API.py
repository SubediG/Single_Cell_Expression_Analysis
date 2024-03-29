import cellxgene_census
import pandas as pd
import scanpy as sc
import math
import tiledb
import anndata
from scipy.stats import ttest_ind
import numpy as np

# interactive node
#srun -n 4 --time=100:00:00 --pty bash -i
#BIGMEM: srun -p bigmem -n 4 --time=100:00:00 --pty bash -i

#my directory
cd active/debruinz_project/gautam_subedi

#opening soma
census = cellxgene_census.open_soma()

# organ = 'string'

# COVID-19 adata with filter
adata_covid = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter= "disease == 'COVID-19'and tissue_general == 'lung' and is_primary_data == True" ,
    column_names = {"obs": ["assay", "cell_type", "tissue", "tissue_general", "disease"]},
)

#Normal adata with same filter
adata_normal = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter= "disease == 'normal'and tissue_general == 'lung' and is_primary_data == True" ,
    column_names = {"obs": ["assay", "cell_type", "tissue", "tissue_general", "disease"]},
)

#Saving adata as h5ad file
output_path1 = "/active/debruinz_project/gautam_subedi/adata_covid.h5ad"
output_path2 = "/active/debruinz_project/gautam_subedi/adata_normal.h5ad"
adata_covid.write(output_path1)
adata_normal.write(output_path2)

#Read anndata
covid_adata = anndata.read_h5ad("/active/debruinz_project/gautam_subedi/adata_covid.h5ad")
normal_adata = anndata.read_h5ad("/active/debruinz_project/gautam_subedi/normal_covid.h5ad")

#Fetching expression matrix
covid_exp_mat = covid_adata.X
normal_exp_mat = normal_adata.X

#cehcking gene_names and cell_type of covid_data
covid_adata.var.feature_name
covid_adata.obs.cell_type

#cehcking gene_names and cell_type of normal_data and covid and non-covid have same feature length and ids
normal_adata.var.feature_name
normal_adata.obs.cell_type

#unique cell_type in covid_adata
adata_covid.obs['cell_type'].unique  #70 categories
adata_normal.obs.feature_name.unique  #60664

## unique cell_type in normal
adata_normal.obs['cell_type'].unique() #184 catrgoreis

# This section shows common expression matrix containing genes and cells that are present in both.

# Common cell_type
unique_cell_types_covid = set(adata_covid.obs['cell_type'].unique())
unique_cell_types_normal = set(adata_normal.obs['cell_type'].unique())
common_cell_types = unique_cell_types_covid.intersection(unique_cell_types_normal)
#print(common_cell_types)  = 66 common cell types

# Output shows that they have same gene name and also at same position with same dimension
covid_feature_names = set(covid_adata.var['feature_name'])
normal_feature_names = set(normal_adata.var['feature_name'])
common_feature_names = covid_feature_names.intersection(normal_feature_names)
#len(common_feature_names)
#60664

gene_names = covid_adata.var.feature_name
covid_gene_data = covid_exp_mat.toarray()
normal_gene_data = normal_exp_mat.toarray()
t_statistics, p_values = ttest_ind(covid_gene_data, normal_gene_data, equal_var=False)
significant_genes_value = p_values < 0.05
significant_gene_names = gene_names.values[significant_genes_value]
significant_t_statistics = t_statistics[significant_genes_value]
significant_p_values = p_values[significant_genes_value]
significant_gene_names =significant_gene_names.astype(str) #Not necessary if saving in dataframe
output_filename = "/active/debruinz_project/gautam_subedi/t_test_results.csv"
df = pd.DataFrame(data)
df.to_csv(output_filename, index=False)


#another approach, This can be revised to use for sparse matrix
#gene_names = covid_adata.var.feature_name
#num_genes = covid_exp_mat.shape[1]
#gene_indices = np.arange(0, num_genes)
#t_statistics = np.zeros(num_genes)
#p_values = np.zeros(num_genes)
#significant_gene_names = []
#significant_t_statistics = []
#significant_p_values = []
#for gene_index in range(num_genes):
#    covid_gene_data = covid_exp_mat[:, gene_index].data
#    normal_gene_data = normal_exp_mat[:, gene_index].data
#    if len(covid_gene_data) >= 2 and len(normal_gene_data) >= 2:
#        t_statistic, p_value = ttest_ind(covid_gene_data, normal_gene_data, equal_var=False)
#        t_statistics[gene_index] = t_statistic
#        p_values[gene_index] = p_value
#        if p_value < 0.05:
#            significant_gene_names.append(gene_names.values[gene_index])
#            significant_t_statistics.append(t_statistic)
#            significant_p_values.append(p_value)
#    else:
#        t_statistics[gene_index] = np.nan
#        p_values[gene_index] = np.nan




