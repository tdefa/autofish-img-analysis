

#%%
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from .spots_detection import remove_double_detection
from tqdm import tqdm
from .utils.segmentation_processing import compute_dico_centroid

if False: # change to True if you want to use imageJ
    None
    import imagej, scyjava
    scyjava.config.add_option('-Xmx40g')
    ij = imagej.init('sc.fiji:fiji')

#https://forum.image.sc/t/grid-collection-stitching-registered-coordinates-not-saving-correctly-with-pyimagej/22942
def stich_with_image_J(
    ij,
STITCHING_MACRO,
    img_pos_dico ,
    stitching_type="[Grid: snake-by-row]",
    order="[Left & Down]",
    grid_size_x =3,
    grid_size_y =1,
    tile_overlap = 10,
    image_name = "r1_pos{i}_ch0.tif",
    image_path = "/media/tom/Transcend/autofish_test_stiching/r1",
    output_path = "/media/tom/Transcend/autofish_test_stiching/r1",):


    ##find the option to save it in a file txt




    #  " image_output=[Write to disk]"+
    ### add compute overlap key word
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for img_name in img_pos_dico:
        print(img_name)
        first_file_index_i = int(img_pos_dico[img_name][0].split('pos')[1])
        print(f'first file index {first_file_index_i}')

        args = {
            "first_file_index_i": first_file_index_i,
            "output_textfile_name": f"TileConfiguration_{img_name}.txt",
            "type": stitching_type,
            "order": order,
            "grid_size_x": grid_size_x,
            "grid_size_y": grid_size_y,
            "tile_overlap": tile_overlap,
            "image_path": image_path,
            "image_name": image_name,
            "output_path": output_path
        }
        ####
        Path(output_path).mkdir(parents=True, exist_ok=True)
        res = ij.py.run_macro(STITCHING_MACRO, args)


###############"" parse the .TXT file generated by imageJ




def parse_txt_file(path_txt =  "/media/tom/T7/Stitch/acquisition/r1_bc1/TileConfiguration.registered_ch1.txt",
                   image_name_regex = "opool1_1_MMStack"):


    file1 = open(path_txt, "r")
    list_line = file1.readlines()

    dico_stitch = {}


    for line in list_line:
        if image_name_regex in line:
            dico_stitch[line.split('; ; ')[0]] = ast.literal_eval(line.split('; ; ')[1])


    ### negative coordinate not allowed

    x_min = 0
    y_min = 0
    z_min = 0
    for img_name in dico_stitch.keys():
        if len(np.array(dico_stitch[img_name])) == 2:
            dico_stitch[img_name] = [dico_stitch[img_name][0], dico_stitch[img_name][1], 0]
        x_min = min(x_min, np.min(np.array(dico_stitch[img_name])[0]))
        y_min = min(y_min, np.min(np.array(dico_stitch[img_name])[1]))
        z_min = min(z_min, np.min(np.array(dico_stitch[img_name])[2]))
    for img_name in dico_stitch.keys():
        dico_stitch[img_name] = np.array(dico_stitch[img_name])
        dico_stitch[img_name] -= np.array([x_min, y_min, z_min])

    dico_stitch_position = {}
    for img_name in dico_stitch.keys():
        position = "pos" + img_name.split("pos")[1].split('_')[0]
        dico_stitch_position[position] = np.array([dico_stitch[img_name][2], dico_stitch[img_name][1], dico_stitch[img_name][0]])
    return dico_stitch_position



def dict_register_artefact(dico_spots_registered,
                           dico_translation,
                       dico_bc_noise= {'r0': 'artefact',
                                'r2': 'artefact'},
                       ref_round = 'r1',
                           list_round = ['r0', 'r1', 'r10', 'r11', 'r12',
                                         'r13', 'r2', 'r3', 'r4',
                                         'r5', 'r6', 'r7', 'r8', 'r9']
                           ):


    dico_spot_artefact = {}
    dico_spot_artefact[ref_round] = {}
    list_position = list(dico_translation.keys())
    for image_position in list_position:
        dico_spot_artefact[ref_round][image_position] = np.concatenate([dico_spots_registered[ra][image_position]
                                                                              for ra in dico_bc_noise] )
    for round in list_round:
        if round == ref_round:
            continue
        dico_spot_artefact[round] = {}
        for image_position in list_position:
            try:
                if round == ref_round:
                    x_translation = 0
                    y_translation = 0
                else:
                    x_translation = -dico_translation[image_position][ref_round][round]['x_translation']
                    y_translation = -dico_translation[image_position][ref_round][round]['y_translation']
                dico_spot_artefact[round][image_position] = dico_spot_artefact[ref_round][image_position].copy() - \
                                                               np.array([0, y_translation, x_translation])
            except KeyError:
               print("no translation for ", round, image_position)
    return dico_spot_artefact

#### stich dico_spots




def stich_dico_spots(dict_spots_registered_df,
                     dict_stitch_img,
                     dict_round_gene = {
                         'r1_bc1': "Rtkn2",
                         'r3_bc4': "Pecam1",
                         'r4_bc5': "Ptprb",
                         'r5_bc6': "Pdgfra",
                         'r6_bc7': "Chil3",
                         'r7_bc3': "Lamp3"
                     },
                     image_shape = [55, 2048, 2048],
                    nb_tiles_x = 3,
                     nb_tiles_y = 3,
                     df_matching_new_cell_label = None ):


    if df_matching_new_cell_label is not None:


        dict_local_global_label = {} # [position][round][old_cell_label][new_cell_label]
        list_position = []
        list_cell_id_local_position = []
        list_cell_id_stitched_mask = []

        for image_name in df_matching_new_cell_label.keys():
            list_position +=  list(df_matching_new_cell_label[image_name]['pos'])
            list_cell_id_local_position += list(df_matching_new_cell_label[image_name]['cell_id_local_position'])
            list_cell_id_stitched_mask += list(df_matching_new_cell_label[image_name]['cell_id_stitched_mask'])

        unique_position = np.unique(list_position)
        for position in unique_position:
            dict_local_global_label[position] = {}

        for index in range(len(list_position)):
            position = list_position[index]
            cell_id_local_position = list_cell_id_local_position[index]
            cell_id_stitched_mask = list_cell_id_stitched_mask[index]
            dict_local_global_label[position][cell_id_local_position] = cell_id_stitched_mask

            dict_local_global_label[position][-1] = -1





    ###  create df coord in ref round + gene
    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    final_shape_xy = np.array([ image_ly * nb_tiles_y + 1000, image_lx * nb_tiles_x + 1000,]) #ten pixels margin
    final_masks = np.zeros([image_shape[0] +10, int(final_shape_xy[0]), int(final_shape_xy[1])], dtype= np.uint16 )
    #### stich all the spots
    list_x = []
    list_y = []
    list_z = []
    list_round = []
    list_gene = []
    list_image_position = []
    list_cell_assignment = []
    dict_spots_registered_stitch_df = {}

    for image_name in dict_stitch_img:
        print(image_name)
        dico_stitch =  dict_stitch_img[image_name]
        for position in dico_stitch:
            print(position)
            if dico_stitch is not None and position in dico_stitch.keys():
                cz, cy, cx = dico_stitch[position]
                cx, cy, cz = round(cx), round(cy), round(cz)
            else:
                cx, cy, cz = 0, 0, 0

            tile_list_x  = dict_spots_registered_df[position]['x']
            tile_list_y = dict_spots_registered_df[position]['y']
            tile_list_z = dict_spots_registered_df[position]['z']
            tile_list_round = dict_spots_registered_df[position]['round']
            if "cell_assignment" in dict_spots_registered_df[position]:
                tile_list_cell_assignment = dict_spots_registered_df[position]['cell_assignment']
                if df_matching_new_cell_label is not None:
                    tile_list_cell_assignment = [dict_local_global_label[position][cell_id_local_position]
                                                 for cell_id_local_position in tile_list_cell_assignment]

            for spot_index in range(len(tile_list_x)):
                if final_masks[int(tile_list_z[spot_index] + cz),
                int(tile_list_y[spot_index] + cy),
                int(tile_list_x[spot_index] + cx)] == 0:
                    list_z.append(tile_list_z[spot_index] + cz)
                    list_y.append(tile_list_y[spot_index] + cy)
                    list_x.append(tile_list_x[spot_index] + cx)
                    list_round.append(tile_list_round[spot_index])
                    list_gene.append(dict_round_gene[tile_list_round[spot_index]])
                    list_image_position.append(position)

                    if "cell_assignment" in dict_spots_registered_df[position]:
                        list_cell_assignment.append(tile_list_cell_assignment[spot_index])
            final_masks[cz : cz +image_shape[0] ,
            cy : cy + image_ly, cx:cx +  image_lx] = np.ones([image_shape[0],image_ly, image_lx ])

        df_coord = pd.DataFrame()
        df_coord['x'] = list_x
        df_coord['y'] = list_y
        df_coord['z'] = list_z
        df_coord['round_name'] = list_round
        df_coord['gene'] = list_gene
        if "cell_assignment" in dict_spots_registered_df[position]:
            df_coord['cell_assignment'] = list_cell_assignment
        df_coord['image_position'] = list_image_position
        dict_spots_registered_stitch_df[image_name] = df_coord

    return dict_spots_registered_stitch_df


    #df_coord.to_csv(f"{args.folder_of_rounds}{args.name_dico}_df_coord.csv", index=False)


#### stich segmask

def stich_segmask(dico_stitch_img, # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                  path_mask = "/media/tom/T7/stich0504/segmentation_mask",
                  path_to_save_mask =  "/media/tom/T7/stich0504/segmentation_mask_stitch",
                  image_shape=[37, 2048, 2048],
                  nb_tiles_x = 3,
                  nb_tiles_y=3,

                  compute_dico_centroid = False,
                  iou_threshold = 0.25):



    Path(path_to_save_mask).mkdir(parents=True, exist_ok=True)
    dict_df_matching_new_cell_label = {}

    for image_name in dico_stitch_img:
        dico_stitch = dico_stitch_img[image_name]
        dico_centroid = {}
        image_lx = image_shape[-2]
        image_ly = image_shape[-1]
        image_lz = image_shape[0]
        max_z = int(np.max(np.array(list(dico_stitch.values()))[:,0]))+1
        final_shape_xy = np.array([ image_ly * nb_tiles_y + 1000, image_lx * nb_tiles_x + 1000]) #ten pixels margin
        final_masks = np.zeros([image_shape[0]+max_z, int(final_shape_xy[0]), int(final_shape_xy[1])], dtype= np.uint16 )
        print(final_masks.shape)
        list_cell_local = []
        list_cell_global = []
        list_pos = []
        for path_ind_mask in tqdm(list(Path(path_mask).glob("*.tif"))[:]):
            image_position = "pos" + path_ind_mask.name.split("pos")[1].split(".")[0].split("_")[0]
            if image_position not in dico_stitch:
                continue
            print(path_ind_mask.name)
            ind_mask = tifffile.imread(path_ind_mask)
            image_position = "pos" + path_ind_mask.name.split("pos")[1].split(".")[0].split("_")[0]
            z_or, y_or, x_or = dico_stitch[image_position]
            x_or, y_or, z_or, = round(x_or), round(y_or), round(z_or)
            ind_mask = ind_mask.astype(np.uint16)
            max_ind_cell =  final_masks.max()
            original_label_list = np.unique(ind_mask)
            ind_mask[ind_mask > 0] = ind_mask[ind_mask > 0] + max_ind_cell
            new_label_list = list(original_label_list + max_ind_cell)
            original_label_list = list(original_label_list)

            #compute iou between mask and final mask
            local_mask  = final_masks[z_or:z_or+image_lz, y_or:y_or + image_ly, x_or:x_or + image_lx]

            if local_mask.shape[0] != ind_mask.shape[0]:
                print("error in the shape of the mask" + path_ind_mask.name)
                missing_z = local_mask.shape[0] - ind_mask.shape[0]
                ind_mask = np.concatenate( [np.zeros([missing_z, local_mask.shape[1], local_mask.shape[1]]), ind_mask]).astype(np.uint16)
                assert local_mask.shape[0] == ind_mask.shape[0]

            present_cell = np.unique(final_masks[:,  y_or:y_or + image_ly, x_or:x_or + image_lx])
            print(f'present_cell {present_cell}')
            if 0 in present_cell:
                present_cell = present_cell[1:]


            for cell in present_cell:
                unique_inter_cell = np.unique(ind_mask[local_mask == cell])
                print(f'unique_inter_cell {unique_inter_cell} , cell {cell}')
                if 0 in unique_inter_cell:
                    unique_inter_cell = unique_inter_cell[1:]

                for inter_cell in np.unique(unique_inter_cell):
                    iou = np.logical_and(ind_mask == inter_cell, local_mask == cell).sum() \
                          / np.logical_or(ind_mask == inter_cell, local_mask == cell).sum()
                    print(f"iou {iou} cell {cell} inter_cell {inter_cell-max_ind_cell}")

                    if iou > iou_threshold:
                        ind_mask[ind_mask == inter_cell] = cell
                        print("iou MATCH ", iou)
                        try:
                            new_label_list[new_label_list.index(inter_cell)] = cell
                        except ValueError:
                            print(f"error in  {inter_cell} already poped")
            list_cell_local += original_label_list
            list_cell_global += new_label_list
            list_pos += [image_position] * len(original_label_list)

            final_masks[z_or:z_or+image_lz, y_or:y_or + image_ly, x_or:x_or + image_lx] = ind_mask
            ###
            if compute_dico_centroid:
                compute_dico_centroid_ind_mask = compute_dico_centroid(mask_nuclei = ind_mask,
                                                                       dico_simu=None,
                                                                       offset=np.array([0, y_or, x_or]))
                for key in compute_dico_centroid_ind_mask.keys():
                    dico_centroid[key] = compute_dico_centroid_ind_mask[key]
                print(f'len dico_centroid {len(dico_centroid)}')
                print()
                (Path(path_to_save_mask) / "dico_centroid").mkdir(parents=True, exist_ok=True)
                np.save((Path(path_to_save_mask) / "dico_centroid") / image_name, dico_centroid)

        df_matching_new_cell_label = pd.DataFrame()
        df_matching_new_cell_label["cell_id_local_position"] = list_cell_local
        df_matching_new_cell_label["cell_id_stitched_mask"] = list_cell_global
        df_matching_new_cell_label["pos"] = list_pos
        dict_df_matching_new_cell_label[image_name] = df_matching_new_cell_label

        np.save(Path(path_to_save_mask) / image_name, final_masks)

        #final_masks[:, x_or:x_or + image_lx, y_or:y_or + image_ly]
    if compute_dico_centroid:

        return  dico_centroid, dict_df_matching_new_cell_label
    else:
        return dict_df_matching_new_cell_label

def stich_from_dico_img(dico_stitch, # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                  path_mask = "/media/tom/Transcend/lustr2023/images/r1_Cy3",
                    regex = "*_ch1*tif*",
                    image_shape=[37, 2048, 2048],
                        nb_tiles_x=3,
                        nb_tiles_y=3,):

    image_lx = image_shape[-2]
    image_ly = image_shape[-1]
    image_lz = image_shape[0]

    final_shape_xy = np.array([image_ly * nb_tiles_y + 1000, image_lx * nb_tiles_x + 1000])  # ten pixels margin
    final_masks = np.zeros([image_shape[0] + 10, int(final_shape_xy[0]), int(final_shape_xy[1])], dtype=np.uint16)
    for path_ind_mask in tqdm(list(Path(path_mask).glob(f"{regex}"))[:]):
        print(f'{path_ind_mask.name} in the folder')

        image_position = "pos" + path_ind_mask.name.split("pos")[1].split(".")[0].split("_")[0]
        if image_position  not in dico_stitch.keys():
            continue
        print(path_ind_mask.name)
        print(f'{path_ind_mask.name} will be stitched')

        try:
            ind_mask = tifffile.imread(path_ind_mask)
            print(path_ind_mask.name, ind_mask.shape)
            image_lz = ind_mask.shape[0]
            z_or, y_or, x_or  = dico_stitch[image_position]
            x_or, y_or, z_or, = round(x_or), round(y_or), round(z_or)
            print(final_masks[z_or:z_or+image_lz, y_or:y_or + image_ly, x_or:x_or + image_lx].shape)
            final_masks[z_or:z_or+image_lz, y_or:y_or + image_ly, x_or:x_or + image_lx] = ind_mask
        except FileNotFoundError:
            print(f"FileNotFoundError {path_ind_mask}")
            continue
    return final_masks


def stich_from_dico_img_folder(dico_stitch_img,
                  path_mask = "/media/tom/Transcend/lustr2023/images/r1_Cy3",
                    regex = "*_ch1*tif*",
                    image_shape=[37, 2048, 2048],
                    nb_tiles = 5):

    for img in dico_stitch_img:
        final_masks = stich_from_dico_img(dico_stitch_img[img],
                  path_mask = path_mask,
                    regex = regex,
                    image_shape=image_shape,
                    nb_tiles = nb_tiles)
        np.save(Path(path_mask) / img, final_masks)




def registered_stich(dico_stitch,
                     dico_translation,
                     ref_round,
                     target_round,
                     ):


    new_dico_stich = {}
    for pos in dico_stitch:
        x_or, y_or, z_or = dico_stitch[pos]
        if target_round != ref_round and target_round in dico_translation[pos][ref_round]:
            x_translation = dico_translation[pos][ref_round][target_round]['x_translation']
            y_translation = dico_translation[pos][ref_round][target_round]['y_translation']
        else:
            x_translation = 0
            y_translation = 0

        new_dico_stich[pos] = [x_or - x_translation, y_or - y_translation, z_or]

    coord  = np.array(list(new_dico_stich.values()))

    min_x = coord[:, 0].min()
    min_y = coord[:, 1].min()
    for pos in new_dico_stich:
        new_dico_stich[pos][0] = new_dico_stich[pos][0] - min_x
        new_dico_stich[pos][1] = new_dico_stich[pos][1] - min_y

    return new_dico_stich, min_x, min_y

#########################" stiched and registered ############################


def register_then_stich(dico_stitch,
                           dico_translation,
                           ref_round = 'r1',
                           target_round  = "r0",
                           path_mask = "/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/r0/",
                           regex = "*_ch0*tif*",
                           image_shape=[40, 2048, 2048],
                           nb_tiles = 5,
                           z_marge = 10,
                           y_marge = 100,
                            x_marge = 100,) :


    image_lz = image_shape[0]
    image_ly = image_shape[1]
    image_lx = image_shape[2]

    final_shape_xy = np.array([image_lx * nb_tiles + y_marge, image_ly * nb_tiles + x_marge]) #ten pixels margin
    final_masks = np.zeros([image_shape[0] + z_marge, int(final_shape_xy[0]), int(final_shape_xy[1])], dtype=np.uint16)
    for path_ind_mask in tqdm(list(Path(path_mask).glob(f"{regex}"))[:]):
        #try:
            print(path_ind_mask.name)
            pos = "pos" + path_ind_mask.name.split("pos")[1].split(".")[0].split("_")[0]
            print(path_ind_mask.name, pos)

            x_translation = dico_translation[pos][ref_round][target_round]['x_translation']
            y_translation = dico_translation[pos][ref_round][target_round]['y_translation']
            x_translation = int(round(x_translation))
            y_translation = int(round(y_translation))
            ind_mask = tifffile.imread(path_ind_mask)
            ########## register the image to R1


            image_lz = ind_mask.shape[0]
            image_ly = ind_mask.shape[1]
            image_lx = ind_mask.shape[2]


            ind_mask_registered = np.zeros(ind_mask.shape, dtype=np.uint16)
            if x_translation > 0 and y_translation > 0:
                ind_mask_registered[:, :-y_translation, :-x_translation] = ind_mask[:, y_translation:,
                                                                                                    x_translation:]
            elif x_translation > 0 and y_translation < 0:
                    ind_mask_registered[:, -y_translation:, :-x_translation] = ind_mask[:, :y_translation,
                                                                                                x_translation:]
            elif x_translation < 0 and y_translation > 0:
                ind_mask_registered[:, :-y_translation, -x_translation:] = ind_mask[:, y_translation:,
                                                                                     :x_translation]
            elif x_translation < 0 and y_translation < 0:
                ind_mask_registered[:, -y_translation:, -x_translation:] = ind_mask[:, :y_translation,
                                                                                     :x_translation]

            elif x_translation == 0 and y_translation > 0:
                ind_mask_registered[:, :-y_translation,:] = ind_mask[:, y_translation:,
                                                               :]
            elif x_translation == 0 and y_translation < 0:
                ind_mask_registered[:, -y_translation:, :] = ind_mask[:, :y_translation,
                                                             :]
            elif x_translation > 0 and y_translation == 0:
                ind_mask_registered[:, :, :-x_translation] = ind_mask[:, :,
                                                             x_translation:]

            elif x_translation < 0 and y_translation == 0:
                ind_mask_registered[:, :, -x_translation:] = ind_mask[:, :,
                                                             :x_translation]

            elif x_translation == 0 and y_translation == 0:
                ind_mask_registered[:, :, :] = ind_mask[:, :, :]


            x_or, y_or, z_or = dico_stitch[pos]
            x_or, y_or, z_or, = round(x_or), round(y_or), round(z_or)
            #if final_masks.shape[0] != ind_mask.shape[0]:
            #    print("error in the shape of the mask" + path_ind_mask.name)
            #    missing_z = final_masks.shape[0] - ind_mask.shape[0]
             #   ind_mask = np.concatenate( [np.zeros([missing_z, ind_mask.shape[1], ind_mask.shape[1]]), ind_mask]).astype(np.uint16)
              #  assert final_masks.shape[0] == ind_mask.shape[0]
            print(final_masks[z_or:z_or+image_lz, y_or:y_or + image_ly, x_or:x_or + image_lx].shape)
            final_masks[z_or:z_or+image_lz, y_or:y_or + image_ly, x_or:x_or + image_lx] = ind_mask_registered


    return final_masks

#%%

if __name__ == "__main__":


    dico_translation = np.load("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_translation.npy",allow_pickle=True).item()


    image_pos7_r1 = tifffile.imread("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/r1/r1_pos7_ch0.tif")


    image_pos7_r5 = tifffile.imread("/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/r0/r0_pos7_ch0.tif")


    x_translation = dico_translation["pos7"]["r1"]['r0'][ 'x_translation']
    y_translation = dico_translation["pos7"]["r1"]['r0'][ 'y_translation']

    ind_mask_registered = np.zeros(image_pos7_r5.shape, dtype=np.uint16)
    #r1 = dico_spots[round_t][image_name] - np.array([0, y_translation, x_translation])
    ind_mask_registered[:, -int(y_translation):, -int(x_translation):] = image_pos7_r5[:, : int(y_translation), :int(x_translation)]

    ind_mask_registered[:, :-int(y_translation), :-int(x_translation)] = image_pos7_r5[:, int(y_translation):, int(x_translation):]


    import napari


    viewer = napari.Viewer()
    viewer.add_image(image_pos7_r1, name = "r1")
    viewer.add_image(image_pos7_r5, name = "r5")
    viewer.add_image(ind_mask_registered, name = "r5 registered")

    ##############################
    #DRAFT
    ##############################


    img_pos_dico = {"img0" : ['pos0', "pos1", "pos2"],
                    "img1" : ['pos6', "pos7", "pos8"]}

    stitching_type = "[Grid: snake-by-row]"
    order = "[Left & Down]"
    grid_size_x =3
    grid_size_y =1
    tile_overlap = 10
    image_name = "r1_pos{i}_ch0.tif"
    image_path = "/media/tom/Transcend/autofish_test_stiching/r1"
    output_path = "/media/tom/Transcend/autofish_test_stiching/r1"
    import imagej
    import scyjava
    scyjava.config.add_option('-Xmx40g')
    ij = imagej.init('sc.fiji:fiji') ## initialize it only once

    for img_name in img_pos_dico:
        print(img_name)
        first_file_index_i = int(img_pos_dico[img_name][0].split('pos')[1])

        stich_with_image_J(
            #ij=ij,
            img_name=img_name,
            stitching_type=stitching_type,
            order=order,
            first_file_index_i=0,
            grid_size_x=3,
            grid_size_y=1,
            tile_overlap=10,
            image_name=image_name,
            image_path=image_path,
            output_path=output_path, )