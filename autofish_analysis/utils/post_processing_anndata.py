



from pathlib import Path
import anndata as ad

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import tifffile
from shapely import geometry, to_geojson, wkt
from shapely.geometry import Polygon
from tqdm import tqdm
from tqdm.auto import tqdm


def mask3D_to_polygone(seg_img_path,
                       cell_id_order):

    if '.tif' in str(seg_img_path):
        seg_img = tifffile.imread(seg_img_path).astype("uint16")
    elif '.npy' in str(seg_img_path)[-5:]:
        seg_img = np.load(seg_img_path)
    else:
        raise ValueError("seg_img_path should be a tif or npy file")



    dico_cell_contours = {}
    for z_index in tqdm(range(seg_img.shape[0])):
        genrator_contours = rasterio.features.shapes(seg_img[z_index])  # rasterio to generate contours
        list_contour = list(genrator_contours)
        for contour in list_contour:
            cell_id = contour[1]
            arr =np.array(contour[0]['coordinates'][0])
            coord =list(np.stack([np.zeros(len(arr)) + z_index, arr[:, 0], arr[:, 1]], axis=1))
            if cell_id not in dico_cell_contours:
                dico_cell_contours[cell_id] = coord
            else:
                dico_cell_contours[cell_id] += coord
    # Convert to shapely Polygons
    polygons = []
    for k_cell in cell_id_order:
        if k_cell in dico_cell_contours:
            polygons.append(Polygon(np.array(dico_cell_contours[k_cell]).astype("uint16")))
        else:
            polygons.append(Polygon())
    return polygons


def get_anndata(dico_gene,
                dico_spots_registered_df,
                cell_column_name = "cell_assignment",
                add_mask_polygone_cell = False,
                path_to_mask_cell = "/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/segmentation_mask",
                round_column_name = "round_name",
                add_mask_polygone_nuclei = False,
                path_to_mask_nuclei = "/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/segmentation_mask",
                ):

    count_matrix_list = []
    batch_index = 0
    batch_index_list = []
    position_list = []
    cell_polygons_list = []
    cell_id_all_pos_list = []
    dico_gene_index = {}
    list_of_df = []

    for r_index in range(len(dico_gene)):
        r = list(dico_gene.keys())[r_index]
        dico_gene_index[dico_gene[r]] = r_index
    for position in tqdm(dico_spots_registered_df):
        print(f"img {position}")
        list_of_dataframes_position = []
        df = dico_spots_registered_df[position]
        df.rename(columns={cell_column_name: "cell"}, inplace=True)
        df.cell =  df.cell.astype(int)

        df["gene"] = [dico_gene[rl] for rl in  df[round_column_name]]
        list_of_dataframes_position.append(df)
        df = pd.concat(list_of_dataframes_position)
        df = df.reset_index(drop=True)


        ### get count matrix
        cell_id_list = []
        for cell, df_cell in df.groupby("cell"):
            if cell != -1:
                cell_id_list.append(cell)
                expression_vector = np.zeros(len(dico_gene))
                for gene, df_gene in df_cell.groupby("gene"):
                    expression_vector[dico_gene_index[gene]] = len(df_gene)
                count_matrix_list.append(expression_vector)
                batch_index_list.append(batch_index)
                position_list.append(position)

        df.cell =  str(batch_index) + '_' + df.cell.astype(str)
        list_of_df.append(df)
        cell_id_all_pos_list += cell_id_list
        if add_mask_polygone_cell == True:
            assert  len(list(Path(path_to_mask_cell).glob(f"*{position}*"))) != 0, f"no mask found for position {position}"
            assert len(list(Path(path_to_mask_cell).glob(f"*{position}*"))) == 1, f"more than one mask found for position {position}"
            seg_img_path = list(Path(path_to_mask_cell).glob(f"*{position}*"))[0]
            cell_polygons = mask3D_to_polygone(seg_img_path = seg_img_path,
                               cell_id_order = cell_id_list)
            cell_polygons_list += cell_polygons
        else:
            cell_polygons_list = None

    ###
    count_matrix = np.array(count_matrix_list)
    cell_polygons_list = [to_geojson(cell_polygons_list[i]) for i in range(len(cell_polygons_list))]


    anndata = ad.AnnData(X=count_matrix,
                         obs=pd.DataFrame({"batch": batch_index_list, "position": position_list,
                                         "cell_id_mask": cell_id_all_pos_list,
                                           "cell_polygons": cell_polygons_list,
                                           }),
                         uns={"points": pd.concat(list_of_df)})

    return anndata


#### extract shapely.geometry.polygon.Polygon


if __name__ == '__main__':


    #import emoji
    seg_img = tifffile.imread("/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/segmentation_mask/r1_pos0_ch0.tif.tif").astype("uint16")

    dico_cell_contours = {}
    for z_index in tqdm(range(seg_img.shape[0])):
        genrator_contours = rasterio.features.shapes(seg_img[z_index])  # rasterio to generate contours
        list_contour = list(genrator_contours)
        for contour in list_contour:
            cell_id = contour[1]
            arr =np.array(contour[0]['coordinates'][0])
            coord =list(np.stack([np.zeros(len(arr)) + z_index, arr[:, 0], arr[:, 1]], axis=1))
            if cell_id not in dico_cell_contours:
                dico_cell_contours[cell_id] = coord
            else:
                dico_cell_contours[cell_id] += coord
    # Convert to shapely Polygons
    import skimage
    from pandas import DataFrame
    from skimage.measure import regionprops_table

    polygons = [Polygon(np.array(dico_cell_contours[k]).astype("uint16")) for k in dico_cell_contours]
    shapes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons))  # Cast to GeoDataFrame
    shapes.drop(
        shapes.area.sort_values().tail(1).index, inplace=True
    )  # Remove extraneous shape
    shapes = shapes[shapes.geom_type != "MultiPolygon"]

    shapes.index = shapes.index.astype(str)


    dico_spots_registered_df.rename(columns={"cell_assignment": "cell"}, inplace=True)


    ## GENERATE ONE ANN


    dico_gene = {'r0': "gene", "r1": "gene1", "r2": "gene2", "r3": "gene3", "r4": "gene4",
                 "r5": "gene5", "r6": "gene6", "r7": "gene7", "r8": "gene8",
                    "r9": "gene9", "r10": "gene10", "r11": "gene11", "r12": "gene12"}


    dico_spots_registered_df = np.load("/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/dict_spots_registered_df_r1_with_cell.npy",
                                       allow_pickle=True).item()
    cell_column_name = "cell_assignment"
    add_mask_polygone_cell = True
    path_to_mask_cell = "/media/tom/Transcend/autofish/2023-07-04_AutoFISH-SABER/segmentation_mask"
    add_mask_polygone_nuclei = False

    dico_round_gene = dico_gene
    ref_round = "r1"
    list_pos = list(dico_spots_registered_df.keys())


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
        df = dico_spots_registered_df[position]
        df.rename(columns={"cell_assignment": "cell"}, inplace=True)
        df.cell =  df.cell.astype(int)

        df["gene"] = [dico_gene[rl] for rl in  df['round']]
        list_of_dataframes_position.append(df)
        df = pd.concat(list_of_dataframes_position)
        df = df.reset_index(drop=True)


        ### get count matrix
        cell_id_list = []
        for cell, df_cell in df.groupby("cell"):
            if cell != -1:
                cell_id_list.append(cell)
                expression_vector = np.zeros(len(dico_gene))
                for gene, df_gene in df_cell.groupby("gene"):
                    expression_vector[dico_gene_index[gene]] = len(df_gene)
                count_matrix_list.append(expression_vector)
                batch_index_list.append(batch_index)
                position_list.append(position)

        df.cell =  str(batch_index) + '_' + df.cell.astype(str)
        list_of_df.append(df)


    ###
    count_matrix = np.array(count_matrix_list)




    import anndata as ad
    anndata = ad.AnnData(X=count_matrix,
                         obs=pd.DataFrame({"batch": batch_index_list, "position": position_list}),
                         uns={"points": pd.concat(list_of_df)})


    anndata.write_h5ad(f"{args.folder_of_rounds}{args.name_dico}_anndata.h5ad")