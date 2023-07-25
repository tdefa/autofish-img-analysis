# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#%%

import argparse

import tifffile

from segmentation import segment_nuclei
from cellpose import models
import numpy as np
#from spots_detection import detection_folder_with_segmentation  # , detection_folder_without_segmentation
from registration import folder_translation
# from stiching import stich_with_image_J, parse_txt_file
import datetime
from pathlib import Path

# Press the green button in the gutter to run the script.

dico_bc_gene0 = {
    'r1_bc1': "Rtkn2",
    'r3_bc4': "Pecam1",
    'r4_bc5': "Ptprb",
    'r5_bc6': "Pdgfra",
    'r6_bc7': "Chil3",
    'r7_bc3': "Lamp3",
    'r8_bc3': "Rtkn2"

}

dico_bc_gene1 = {
    'r1': "Rtkn2",
    'r2': "Lamp3",
    'r3': "Pecam1",
    'r4': "Ptprb",
    'r5': "Pdgfra",
    'r6': "Chil3",
    'r7': "Apln",
    "r8": "Fibin",
    "r9": "Pon1",
    "r10": "Cyp2s1",
    "r11": "C3ar1",
    "r12": "Hhip",
    #"r13": "Hhip",
}


dico_bc_noise = {"r1_Cy5": "artefact"}


if __name__ == '__main__':

    ####### segment individual tile

    # input folder name + regex_file_name
    # result a new save folder with the same name + _segmented

    parser = argparse.ArgumentParser(description='test')

    parser.add_argument("--folder_of_rounds",
                        type=str,
                        default="/media/tom/Transcend/autofish_test/",
                        help='')

    parser.add_argument("--path_to_dapi_folder",
                        type=str,
                        default="/media/tom/Transcend/autofish_test/r1/",
                        help='')

    parser.add_argument("--path_to_mask_dapi",
                        type=str,
                        default="/media/tom/Transcend/autofish/2023-06-28_AutoFISH_22rounds/segmentation_mask/",
                        help='')
    parser.add_argument("--regex_dapi",
                        type=str,
                        default="ch0",
                        help='')

    parser.add_argument("--fixed_round_name",
                        type=str,
                        default="r1",
                        help='')
    parser.add_argument("--folder_regex_round",
                        type=str,
                        default="r",
                        help='')
    parser.add_argument("--chanel_regex",
                        type=str,
                        default="ch0",
                        help='channel for the fish')
    parser.add_argument("--image_name_regex",
                        type=str,
                        default="r",
                        help='')

    parser.add_argument("--name_dico",
                        type=str,
                        default="ref_r3",
                        help='')

    ### param for detection
    parser.add_argument("--local_detection",
                        type=int,
                        default=0,
                        help='')

    parser.add_argument("--use_median_threshold",
                        type=int,
                        default=0,
                        help='')
    parser.add_argument("--mask_artefact", type=int, default=0)
    parser.add_argument("--artefact_filter_size", type=int, default=32)
    parser.add_argument("--sigma_detection", type=float, default=1.3)
    parser.add_argument("--remove_non_sym", type=int, default=0)

    ######## parameters stiching
    parser.add_argument("--image_shape",
                        type=list,
                        default=[38, 2048, 2048],
                        help='')

    parser.add_argument("--nb_tiles",
                        type=int,
                        default=5,
                        help='')

    parser.add_argument("--regex_image_stiching",
                        type=str,
                        default="r1_pos{i}_ch0.tif",
                        )

    parser.add_argument("--remove_double_detection", type = int, default = 0)
    parser.add_argument("--stich_segmentation_mask", type = int, default = 1)

    ##### task to do
    parser.add_argument("--NDTiffStack_to_tiff", default=0, type=int)

    parser.add_argument("--segmentation", default=0, type=int)
    parser.add_argument("--registration", default=1, type=int)

    parser.add_argument("--spots_detection", default=1, type=int)
    parser.add_argument("--signal_quality", default=0, type=int)
    parser.add_argument("--spots_registration", default=1, type=int)

    parser.add_argument("--stitch", default=0, type=int)
    parser.add_argument("--stich_spots_detection", default=0, type=int)
    parser.add_argument("--plot_control_quality", default=0, type=int)
    parser.add_argument("--generate_comseg_input", default=0, type=int)

    parser.add_argument("--port", default=39948)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--host", default='127.0.0.2')

    args = parser.parse_args()

    print(args)


    args.image_path_stiching = args.folder_of_rounds + args.fixed_round_name
    args.output_path_stiching = args.folder_of_rounds + args.fixed_round_name + "output_s"

    e = datetime.datetime.now()
    date_str = f"{e.month}_{e.day}_{e.hour}_{e.minute}_{e.second}"
    #%%

    ####################
    ####### segment individual tile
    ####################


    ## NDTiffStack to img folder

    if args.NDTiffStack_to_tiff == 1:

        from utils.split_ndtiff_stack import NDTiffStack_to_tiff

        NDTiffStack_to_tiff(

            path_parent=Path(r'/media/tom/Transcend/2023-07-04_AutoFISH-SABER/'),
            # path_parent = Path(r'/Users/fmueller/Documents/data')
            n_z=33,  # number of z plances
            n_pos=3,  # number of field of images

            # Number of channels
            n_c_default=1,
            n_c_dict={'r0_1': 2, 'r1_1': 2},
            n_c_keys=list(n_c_dict.keys()),

            # Diverse string replacements
            string_replacements_path=[('images_multi-stack', 'images'),
                                      ('_1', '')],

            string_replacements_file=[('_NDTiffStack.*', ''),
                                      ("_bgd*", ""),
                                      ('Pos', 'pos'),
                                      ],
        )








    if args.segmentation == 1:
        from segmentation import segment_nuclei
        from cellpose import models
        from pathlib import Path

        print("segmentation")

        dico_param = {}
        dico_param["diameter"] = 230
        dico_param["flow_threshold"] = 0.6
        dico_param["mask_threshold"] = 0
        dico_param["do_3D"] = False
        dico_param["mip"] = False
        dico_param["projected_focused"] = False
        dico_param["stitch_threshold"] = 0.3
        dico_param["erase_solitary"] = True
        dico_param["erase_small_nuclei"] = 300
        dico_param["model_type"] = "cyto"
        dico_param["gpu"] = True



        segment_nuclei(path_to_staining=args.path_to_dapi_folder,
                       regex_dapi=args.regex_dapi,
                       path_to_mask_dapi=args.path_to_mask_dapi,
                       dico_param=dico_param,
                       save=True,
                       )

    ########
    # register each channel to the dapi round channel
    #######

    if args.registration == 1:


        dico_translation = folder_translation(folder_of_rounds=args.folder_of_rounds,  # works ok
                                              fixed_round_name=args.fixed_round_name,
                                              folder_regex=args.folder_regex_round,
                                              chanel_regex=args.chanel_regex,
                                              registration_repeat=5)

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy", dico_translation)

    ####################
    ## Do individual spots detection (optional : using mask segmentation)
    ####################
    if args.spots_detection == 1:
        print("spots detection")
        if args.local_detection:
            dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                       allow_pickle=True).item()
        else:
            dico_translation = None

        if args.mask_artefact:
            dico_spot_artefact = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spot_artefact.npy",
                                       allow_pickle=True).item()
            artefact_filter_size = args.artefact_filter_size
        else:
            dico_spot_artefact = None
            artefact_filter_size = None
        threshold_input = None
        dico_spots, dico_threshold = detection_folder_with_segmentation(
            round_folder_path=args.folder_of_rounds,
            round_name_regex=args.folder_regex_round,
            image_name_regex=args.image_name_regex,
            channel_name_regex=args.chanel_regex,
            fixed_round_name=args.fixed_round_name,
            path_output_segmentaton=args.path_to_mask_dapi,
            min_distance=(4, 4, 4),
            scale_xy=0.108,
            scale_z=0.300,
            sigma=args.sigma_detection,
            ## mask artefact
            dico_spot_artefact=dico_spot_artefact,
            artefact_filter_size = artefact_filter_size,
            ### detection parameters with segmentation
            dico_translation=dico_translation,
            diam_um=20,
            local_detection=args.local_detection,
            min_cos_tetha=0.70,
            order=5,
            test_mode=False,
            threshold_merge_limit=0.330,
            file_extension='tif',
            threshold_input=threshold_input,
                use_median_threshold = args.use_median_threshold,
            remove_non_sym=args.remove_non_sym,
        )

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}" + \
                f"_{args.folder_regex_round}_mask_artefact{args.mask_artefact}_{args.sigma_detection}_remove_non_sym{args.remove_non_sym}.npy",
                dico_spots)

    ########
    # Compute signal quality for each round
    #######

    if args.signal_quality == 1:

        from utils.signal_quality import compute_quality_all_rounds



        dico_spots = np.load \
            (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}_{args.folder_regex_round}.npy",
             allow_pickle=True).item()


        ######################" add the automatic plot


        dico_signal_quality = compute_quality_all_rounds(
            dico_spots = dico_spots,
            round_folder_path=args.folder_of_rounds,
            round_name_regex=args.folder_regex_round,
            image_name_regex=args.image_name_regex,
            channel_name_regex=args.chanel_regex,
            file_extension="tif",
            voxel_size=[300, 108, 108],
            spot_radius=300,
            sigma=1.3,
            order=5,
            compute_sym=True,
            return_list=True,
        )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_signal_quality{args.local_detection}.npy", dico_signal_quality)

        for round_str in dico_signal_quality:
            mean_intensity = []
            mean_snr = []
            mean_symmetry_coef = []
            mean_background = []
            for image in dico_signal_quality[round_str]:
                mean_intensity += dico_signal_quality[round_str][image]["intensity"]
                mean_background += dico_signal_quality[round_str][image]["background"]
                mean_snr += dico_signal_quality[round_str][image]["snr"]
                mean_symmetry_coef += dico_signal_quality[round_str][image]["symmetry_coef"]

            print(f"round {round_str} mean intensity {np.mean(mean_intensity)}")
            print(f"round {round_str} median intensity {np.median(mean_intensity)}")
            print()
            print(f"round {round_str} mean background {np.mean(mean_background)}")
            print(f"round {round_str} median background {np.median(mean_background)}")
            print()

            print(f"round {round_str} mean snr {np.mean(mean_snr)}")
            print(f"round {round_str} median snr {np.median(mean_snr)}")
            print()

            print(f"round {round_str} mean symmetry coef {np.mean(mean_symmetry_coef)}")
            print(f"round {round_str} median symmetry coef {np.median(mean_symmetry_coef)}")

            print()
            print()
            print()

    ########
    # Stitch only ref round images
    #######

    ########################### Registration of the spots ################
    if args.spots_registration == 1:
        from stitching import spots_registration

        dico_spots = np.load \
            (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}" + \
             f"_{args.folder_regex_round}_mask_artefact{args.mask_artefact}_{args.sigma_detection}_remove_non_sym{args.remove_non_sym}.npy",
             allow_pickle=True).item()
        dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                   allow_pickle=True).item()
        dico_spots_registered_df, dico_spots_registered, missing_data = spots_registration(
            dico_spots,
            dico_translation,
            ref_round=args.fixed_round_name,
            check_removal=False,
            threshold_merge_limit=0.330,
            scale_xy=0.103,
            scale_z=0.270,
        )

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}" , dico_spots_registered_df)
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_{args.fixed_round_name}" , dico_spots_registered)



    ##############
    #  add nucleus label or cell label to the spots using the non stiched label
    #############

    if args.add_label_to_spots == 1:
        from registration import spots_assignement
        dico_spots_registered_df = spots_assignement(dico_spots_registered_df,
                          path_to_masks=args.path_to_masks,
                          files_mask_extension="*.tif*", )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}" , dico_spots_registered_df)

    ########
    # Stitch detected spots
    #######

    if args.stitch == 1:
        from stitching import stich_with_image_J, parse_txt_file
        from utils.macro import STITCHING_MACRO

        img_pos_dico = {"img0": ['pos0', "pos1", "pos2"],
                        "img1": ['pos6', "pos7", "pos8"]}

        stitching_type = "[Grid: snake-by-row]"
        order = "[Left & Down]"
        grid_size_x = 3
        grid_size_y = 1
        tile_overlap = 10
        image_name = "r1_pos{i}_ch0.tif"
        image_path = "/media/tom/Transcend/autofish_test_stiching/r1"
        output_path = "/media/tom/Transcend/autofish_test_stiching/r1"

        import imagej, scyjava
        scyjava.config.add_option('-Xmx40g')
        ij = imagej.init('sc.fiji:fiji')

        for img_name in img_pos_dico:
            print(img_name)
            first_file_index_i = int(img_pos_dico[img_name][0].split('pos')[1])
            Path(output_path).mkdir(parents=True, exist_ok=True)
            STITCHING_MACRO
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
            res = ij.py.run_macro(STITCHING_MACRO, args)

        from stitching import parse_txt_file

        dico_stitch_img = {}
        for img_name in img_pos_dico:
            path_txt = Path(output_path) / f"TileConfiguration_{img_name}.txt"
            dico_stitch = parse_txt_file \
                (path_txt=path_txt,
                 image_name_regex="_pos", )
            dico_stitch_img[img_name] = dico_stitch
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_stitch_img.npy", dico_stitch_img)


        ############ point stiching
        from stitching import stich_dico_spots

        dico_spots_registered = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_{args.fixed_round_name}.npy",
                                        allow_pickle=True).item()
        dico_spots_registered_df =  np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}.npy",
                                        allow_pickle=True).item()

        dico_stitch_img = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_stitch_img.npy",
                                  allow_pickle=True).item()

        #dico_spots_registered = np.load(f"/media/tom/Transcend/autofish_test_stiching/dico_spots_registered_r1.npy",
        #                                allow_pickle=True).item()


        #dico_stitch_img = np.load(f"/media/tom/Transcend/autofish_test_stiching/dico_stich_img.npy",
        #                          allow_pickle=True).item()



        dico_spots_registered_stitch_df = stich_dico_spots(
            dico_spots_registered = dico_spots_registered,
                             dico_stitch_img = dico_stitch_img,
                             dico_bc_gene={
                                 'r1': "gene1",
                                 'r5': "gene2",
                             },
                             image_shape=[38, 2048, 2048],
                         nb_tiles_x=3,
                         nb_tiles_y=1,
                         )

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_stitch_df.npy",
                                  allow_pickle=True).item()



        ########## image stitching seg mask
        from stitching import stich_segmask

        path_mask = "/media/tom/Transcend/autofish_test_stiching/segmentation_mask"
        path_to_save_mask = "/media/tom/Transcend/autofish_test_stiching/segmentation_mask_stitch"

        _, dico_centroid, df_matching_new_cell_label = stich_segmask(dico_stitch_img,
                                                                               # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                                                                               path_mask=path_mask,
                                                                               path_to_save_mask=path_to_save_mask,
                                                                               image_shape=args.image_shape,
                                                                               nb_tiles_x=3,
                                                                               nb_tiles_y=1,

                                                                               compute_dico_centroid=True,
                                                                               iou_threshold=0.10)

        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_centroid.npy", dico_centroid)
        #np.save(f"{args.folder_of_rounds}{args.name_dico}_df_matching_new_cell_label.npy", df_matching_new_cell_label)
        df_matching_new_cell_label.to_csv(f"{args.folder_of_rounds}{args.name_dico}_df_matching_new_cell_label.csv")


        ######### fish signal stiching
        from stitching import stich_from_dico_img

        image_shape = [38, 2048, 2048]
        nb_tiles_x = 3
        nb_tiles_y = 1
        for image_name in dico_stitch_img:
            dico_stitch = dico_stitch_img[image_name]
            final_masks = stich_from_dico_img(dico_stitch,
                                              # np.load(f"/media/tom/T7/Stitch/acquisition/2mai_dico_stitch.npy",allow_pickle=True).item()
                                              path_mask="/media/tom/Transcend/autofish_test_stiching/r1",
                                              regex="*_ch0*tif*",
                                              image_shape=image_shape,
                                              nb_tiles_x=nb_tiles_x,
                                              nb_tiles_y=nb_tiles_y, )

            np.save(
                Path("/media/tom/Transcend/autofish_test_stiching/segmentation_mask_stitch/stitch_input") / image_name,
                final_masks)


        #### generate registration dico
        print(" you need to mannualy modify the path to the registered file in the txt file")
        dico_stitch = parse_txt_file \
            (path_txt="/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/r1_bc1/TileConfiguration0407_Bis.registered.txt",
             image_name_regex="_pos", )
        np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_stitch.npy",
                dico_stitch)

        if args.stich_spots_detection:  ## get a dataframe with the spots codinates in the ref round

            from stitching import stich_segmask,stich_dico_spots

            dico_stitch = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_stitch.npy",
                allow_pickle=True).item()


            dico_spots = np.load \
                (f"{args.folder_of_rounds}{args.name_dico}_dico_spots_local_detection{args.local_detection}" + \
                    f"_{args.folder_regex_round}_mask_artefact{args.mask_artefact}_{args.sigma_detection}_remove_non_sym{args.remove_non_sym}.npy",
                 allow_pickle=True).item()
            dico_translation = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_translation.npy",
                                       allow_pickle=True).item()

            dico_bc_gene2 = {'r1': 'Rtkn2',
                             'r3': 'Pecam1',
                             'r4': 'Ptprb',
                             'r5': 'Pdgfra',
                            # 'r6': 'Chil3',
                             'r7': 'Apln',
                             'r12': 'Hhip'}

            dico_bc_gene0 = {
             #   'r1_bc1': "Rtkn2",
                'r3_bc4': "Pecam1",
                'r4_bc5': "Ptprb",
                'r5_bc6': "Pdgfra",
                'r6_bc7': "Chil3",
                'r7_bc3': "Lamp3",
                'r8_bc1': "Rtkn2"

            }


            dico_bc_gene2 = {'r2': 'chill3'}



            df_coord, new_spot_list_dico, missing_data, dico_spots_registered = stich_dico_spots(dico_spots = dico_spots,
                                                                          dico_translation = dico_translation,
                                                                          dico_stitch = dico_stitch,
                                                                          ref_round=args.fixed_round_name,
                                                                          dico_bc_gene=dico_bc_gene0,
                                                                          image_shape=args.image_shape,
                                                                          nb_tiles=args.nb_tiles,
                                                                          check_removal = False,
                                                                          threshold_merge_limit=0.330,
                                                                          scale_xy=0.108,
                                                                          scale_z=0.300,
                                                                          )


            if args.remove_double_detection:
                from stitching import dico_register_artefact
                dico_bc_noise = {'r1': 'artefact'
                                    }
                df_coord_noise, _, _, dico_spots_registered = stich_dico_spots(dico_spots = dico_spots,
                                                                              dico_translation = dico_translation,
                                                                              dico_stitch = dico_stitch,
                                                                              ref_round=args.fixed_round_name,
                                                                              dico_bc_gene=dico_bc_noise,
                                                                              image_shape=args.image_shape,
                                                                              nb_tiles=args.nb_tiles,
                                                                                check_removal = False,
                                                                            )
                np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered.npy", dico_spots_registered)
                ## get artefact spots array
                dico_spot_artefact = dico_register_artefact(dico_spots_registered,
                                        dico_translation,
                                    dico_bc_noise= dico_bc_noise,
                                    ref_round = args.fixed_round_name,
                                    list_round = ['r0', 'r1', 'r10', 'r11', 'r12',
                                                 'r13', 'r2', 'r3', 'r4',
                                                 'r5', 'r6', 'r7', 'r8', 'r9']
                                    )

                np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spot_artefact.npy", dico_spot_artefact)
                ## generate artefact mask for ref round
                ## register the artefact position to all other round
                dico_spots = np.load("/media/tom/T7/plane_03/plane_03/20juin_dico_spots_local_detection0_r_mask_artefact0_1.1.npy", allow_pickle=True).item()
                ## remove artefact spots from all other round
                from sklearn.neighbors import NearestNeighbors
                X_noise = np.array(list(zip(df_coord_noise.z, df_coord_noise.y, df_coord_noise.x)))
                X = np.array(list(zip(df_coord.z, df_coord.y, df_coord.x)))
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_noise)

                print("before removing double detection", len(df_coord))
                distances, indices = nbrs.kneighbors(X)

                distances = np.min(distances, axis = 1)
                print("after removing double detection", len(df_coord[distances > 6]))

                df_coord = df_coord[distances > 6]

                df_coord.index = range(len(df_coord))
                print("after removing double detection", len(df_coord))
            ########
            # Stitch stich_segmentation_mask
            #######

            if args.stich_segmentation_mask:
                from stitching import  stich_segmask

                final_masks, dico_centroid, dico_new_to_old_label_pos = stich_segmask(dico_stitch,
                                            path_mask=args.path_to_mask_dapi,
                                            image_shape=args.image_shape,
                                            nb_tiles=args.nb_tiles,
                                            compute_dico_centroid = True)
                np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_new_to_old_label_pos.npy", dico_new_to_old_label_pos)

                np.save(f"{args.folder_of_rounds}{args.name_dico}_final_masks.npy", final_masks)

                np.save(f"{args.folder_of_rounds}{args.name_dico}_final_masks_mip.npy", np.amax(final_masks, 0))
                tifffile.imsave(f"{args.folder_of_rounds}{args.name_dico}_final_masks_mip.tif", np.amax(final_masks, 0))

                np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_nuclei_centroid.npy", dico_centroid)


                from utils.segmentation_processing import compute_dico_centroid

                dico_nuclei_centroid = compute_dico_centroid(mask_nuclei = final_masks)

                np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_nuclei_centroid.npy", dico_nuclei_centroid)


                del final_masks

                from stitching import stich_from_dico

                final_dapi = stich_from_dico(dico_stitch,
                                path_mask="/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/r1_bc1/",
                                regex="*_ch3*tif*",
                                image_shape=[55, 2048, 2048],
                                nb_tiles=9)
                np.save(f"{args.folder_of_rounds}{args.name_dico}_final_dapi.npy", final_dapi)

                np.save(f"{args.folder_of_rounds}{args.name_dico}_final_dapi_mip.npy", np.amax(final_dapi, 0))
                tifffile.imsave(f"{args.folder_of_rounds}{args.name_dico}_final_dapi_mip.tif", np.amax(final_dapi, 0))

                if False: ## plot all stiched round

                    from stitching import registered_stich, stich_from_dico
                    from tqdm import tqdm

                    dico_stitch = np.load('/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_stitch.npy',
                                          allow_pickle=True).item()
                    dico_translation = np.load(
                        '/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_translation.npy',
                        allow_pickle=True).item()
                    ref_round = "r1"
                    path_to_saved_stich = "/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/stiched/"
                    Path(path_to_saved_stich).mkdir(exist_ok=True, parents=True)
                    dico_translation_stich = {}
                    for target_round in tqdm(list(dico_translation["pos0"][ref_round].keys())[7:]):
                        print(target_round)
                        new_dico_stich, min_x, min_y = registered_stich(dico_stitch,
                                                                        dico_translation,
                                                                        ref_round,
                                                                        target_round,
                                                                        )
                        dico_translation_stich[target_round] = [min_x, min_y]
                        final = stich_from_dico(new_dico_stich,
                                                path_mask=f"/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/{target_round}/",
                                                regex="*_ch0*tif*",
                                                image_shape=[40, 2048, 2048],
                                                nb_tiles=5)
                        np.save(f"{path_to_saved_stich}{target_round}.npy", final)
                        np.save(f"{path_to_saved_stich}{target_round}_mip.npy", np.amax(final, 0))
                        tifffile.imsave(f"{path_to_saved_stich}{target_round}_mip.tif", np.amax(final, 0))

                if False: ## plot all stiched round registred


                    from stitching import register_then_stich, stich_from_dico
                    from tqdm import tqdm

                    dico_stitch = np.load('/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_stitch.npy',
                                          allow_pickle=True).item()
                    dico_translation = np.load(
                        '/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/14juin_dico_translation.npy',
                        allow_pickle=True).item()

                    ref_round = "r1"
                    path_to_saved_stich = "/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/stiched_registered/"
                    Path(path_to_saved_stich).mkdir(exist_ok=True, parents=True)
                    for target_round in tqdm(list(dico_translation["pos0"][ref_round].keys())[:7]):
                        print(target_round)
                        final = register_then_stich(dico_stitch,
                                               dico_translation,
                                               ref_round='r1',
                                               target_round=target_round,
                                               path_mask="/media/tom/T7/2023-09-06_LUSTRA-14rounds/image_per_pos/" + target_round + "/",
                                               regex="*_ch0*tif*",
                                               image_shape=[40, 2048, 2048],
                                               nb_tiles=5,
                                               )


                        np.save(f"{path_to_saved_stich}{target_round}.npy", final)
                        np.save(f"{path_to_saved_stich}{target_round}_mip.npy", np.amax(final, 0))
                        tifffile.imsave(f"{path_to_saved_stich}{target_round}_mip.tif", np.amax(final, 0))



                ########
                # Genrate comseg imput
                #######
            if args.generate_comseg_input:

                    final_masks = np.load(f"{args.folder_of_rounds}{args.name_dico}_final_masks.npy")
                    x_list = list(df_coord.x)
                    y_list = list(df_coord.y)
                    z_list = list(df_coord.z)
                    nuc_prior = []
                    in_nuc = []
                    not_in_nuc = []
                    for ix in range(len(z_list)):
                        zz = np.min([int(z_list[ix]), 39])
                        nuc_index_prior = final_masks[zz, int(y_list[ix]), int(x_list[ix])]
                        nuc_prior.append(nuc_index_prior)
                        if nuc_index_prior != 0:
                            in_nuc.append([int(y_list[ix]), int(x_list[ix])])
                        else:
                            not_in_nuc.append([int(y_list[ix]), int(x_list[ix])])

                    df_coord["in_nucleus"] = nuc_prior

                    dico_dico_commu = {"stich0": {"df_spots_label": df_coord, }}
                    np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_dico_commu_all_pos.npy", dico_dico_commu)

                    #np.save(f"/media/tom/Transcend/lustr2023/dico_dico_commu_1juin_global_detect", dico_dico_commu)


                    df_coord = df_coord[np.isin(df_coord.image_position, ['pos0', "pos1", "pos3", "pos4", "pos6",  "pos7"])]
                    df_coord = df_coord.reset_index(drop=True)
                    dico_dico_commu = {"stich0": {"df_spots_label": df_coord, }}
                    np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_dico_commu_mask_removal_6pos.npy", dico_dico_commu)




###########################################################
# geneate a adatan like bento https://bento-tools.readthedocs.io/en/latest/api.html#module-bento.ds
###########################################################


        if args.generate_andata == 1:
            np.save(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}",
                    dico_spots_registered_df)


        dico_spots_registered_df = np.load(f"{args.folder_of_rounds}{args.name_dico}_dico_spots_registered_df_{args.fixed_round_name}.npy",

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




                ### get count matrix

                ### bartch list


                ### update cell index dataframe add position and round







    df = pd.concat(list_of_dataframes)





        if args.plot_control_quality:

            ### check registration

            Path(args.folder_of_rounds +args.ref_round).glob()


            import napari
            import tifffile
            import numpy as np

            image1 = tifffile.imread("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/r1_bc1/r1_bc1_pos4_ch1.tif")
            image2 = tifffile.imread("/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/r4_bc5/r4_bc5_pos4_ch1.tif")

            image1 = np.amax(image1, 0)
            image2 = np.amax(image2, 0)

            dico_translation = np.load(f"/media/tom/T7/2022-02-24_opool-1/Stitch/analyse_paper/29juin_dico_translation.npy",
                                       allow_pickle=True).item()

            x_translation = dico_translation['pos4']['r1_bc1']['r4_bc5']['x_translation']
            y_translation = dico_translation['pos4']['r1_bc1']['r4_bc5']['y_translation']

            from registration import shift
            shifted_image4 = shift(image2, translation=(x_translation, y_translation))


            viewer = napari.view_image(image1, name="image1")
            viewer.add_image(image2, name="image2")

            viewer.add_image(shifted_image4, name="image_2_registered")


            ########### try histogram matching

        ###############" add nuclei prior to spots detection

    #color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #viewer.add_points(in_nuc, name=f"in_nuc", face_color=color, edge_color=color, size=6)

    #color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    #viewer.add_points(not_in_nuc, name=f"not_in_nuc", face_color=color, edge_color=color, size=6)

    ################ compute registration accross round

    ### input folder of folder with the rounds, regex to fish images to take into account

    ### output dictionary with dico[image_name][moving_image][static_round_image] = translation s.th. moving_image + translation = static image
    ### static image is typically round one here

    ################### perform spots detection wit
