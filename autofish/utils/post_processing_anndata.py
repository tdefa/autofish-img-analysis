



import pandas as pd
import numpy as np
import anndata as ad
from tqdm import tqdm

def get_anndata(dico_round_gene,
                dico_spots_registered_df,
                cell_column_name = "cell_assignment",):

    first_round = list(dico_round_gene.keys())[0]
    list_pos = list(dico_spots_registered_df[first_round])
    list_of_df = []
    count_matrix_list = []
    batch_index = 0
    batch_index_list = []
    position_list = []
    dico_gene_index = {}
    list_of_df = []
    for r_index in range(len(dico_round_gene)):
        r = list(dico_round_gene.keys())[r_index]
        dico_gene_index[dico_round_gene[r]] = r_index
    for position in tqdm(list_pos):
        list_of_dataframes_position = []
        for round in dico_spots_registered_df:
            df = dico_spots_registered_df[round][position]
            df.rename(columns={cell_column_name: "cell"}, inplace=True)
            df.cell =  str(batch_index) + '_' + df.cell.astype(str)
            df["gene"] = dico_round_gene[round]
            list_of_dataframes_position.append(df)
        df = pd.concat(list_of_dataframes_position)
        df = df.reset_index(drop=True)
        list_of_df.append(df)


        ### get count matrix
        for cell, df_cell in df.groupby("cell"):
            expression_vector = np.zeros(len(dico_gene_index))
            for gene, df_gene in df_cell.groupby("gene"):
                expression_vector[dico_gene_index[gene]] = len(df_gene)
            count_matrix_list.append(expression_vector)
            batch_index_list.append(batch_index)
            position_list.append(position)
    count_matrix = np.array(count_matrix_list)
    anndata = ad.AnnData(X=count_matrix,
                         obs=pd.DataFrame({"batch": batch_index_list, "position": position_list}),
                         uns={"points": pd.concat(list_of_df)})

    return anndata

if __name__ == '__main__':
    dico_spots_registered_df = np.load \
        (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}.npy",

                                       allow_pickle=True).item()

    dico_spots_registered_df.rename(columns={"cell_assignment": "cell"}, inplace=True)


    ## GENERATE ONE ANN


    dico_gene = {"r1": "gene1", "r2": "gene2", "r3": "gene3",
                 "r5": "gene5"}

    ref_round = args.fixed_round_name
    list_pos = list(dico_spots_registered_df[ref_round].keys())

    count_matrix_list = []
    batch_index = 0
    batch_index_list = []
    position_list = []
    dico_gene_index = {}
    list_of_df = []
    for r_index in range(len(dico_gene)):
        r = list(dico_gene.keys())[r_index]
        dico_gene_index[dico_gene[r]] = r_index
    for position in tqdm(list_pos):
        list_of_dataframes_position = []
        for round in dico_spots_registered_df:
            df = dico_spots_registered_df[round][position]
            df.rename(columns={"cell_assignment": "cell"}, inplace=True)
            df.cell =  str(batch_index) + '_' + df.cell.astype(str)
            df["gene"] = dico_gene[round]
            list_of_dataframes_position.append(df)
        df = pd.concat(list_of_dataframes_position)
        df = df.reset_index(drop=True)
        list_of_df.append(df)


        ### get count matrix
        for cell, df_cell in df.groupby("cell"):
            expression_vector = np.zeros(len(dico_gene))
            for gene, df_gene in df_cell.groupby("gene"):
                expression_vector[dico_gene_index[gene]] = len(df_gene)
            count_matrix_list.append(expression_vector)
            batch_index_list.append(batch_index)
            position_list.append(position)
    ###
    count_matrix = np.array(count_matrix_list)


    import anndata as ad
    anndata = ad.AnnData(X=count_matrix,
                         obs=pd.DataFrame({"batch": batch_index_list, "position": position_list}),
                         uns={"points": pd.concat(list_of_df)})


    anndata.write_h5ad(f"{args.folder_of_rounds}{args.name_dico}_anndata.h5ad")