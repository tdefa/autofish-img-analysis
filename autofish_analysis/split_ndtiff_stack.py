# PREPROCESSING SCRIPT
# Images are stored as multi-channel images. During image acquisition one additional image
# is stored with the LED off. This image has to be removed.


import re
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm


def NDTiffStack_to_tiff(

    path_parent = Path(r'/media/tom/Transcend/autofish_test_input/'),
    n_z = 33,  # number of z plances
    n_pos = 3,  # number of field of images
    # Number of channels
    n_c_default = 1,
    n_c_dict = {'r0_1': 2, 'r1_1': 2},
    # Diverse string replacements
    string_replacements_path = [('images_multi-stack', 'images'),
                                ('_1', '')],

    string_replacements_file = [('_NDTiffStack.*', ''),
                                ("_bgd*", ""),
                                ('Pos', 'pos'),
                               ],
    folder_save = "/media/tom/Transcend/autofish_test/"
        ):

    """
    Args:
        path_parent: (Str) Path to main folder containing the round folder of NDTiffStack
        n_z: (int) number of z stack
        n_pos: (int) number of field of images
        n_c_default: (int)  default number of channel
        n_c_dict: (dict) add round that have a different number of channel like additional DAPI staining e
        ex :{'r0_1': 2, 'r1_1': 2}
        string_replacements_path: (list of tuple) list of tuple of string to replace in the folder name
        string_replacements_file: (list of tuple) list of tuple of string to replace in the file name
        folder_save: (str) path to the  folder to save the tiff
    Returns:
    list of exception if any
    """
    n_c_keys = list(n_c_dict.keys())
    path_parent = Path(path_parent)
    folder_save = Path(folder_save)
    folder_save.mkdir(parents=True, exist_ok=True)
    path_list = []
    for child in path_parent.glob('**/'):
        # for child in path_parent.rglob('*'):
        path_list.append(child)
    path_list.remove(path_parent)
    print(path_list)
    # %% Loop over all subfolders
    list_exceptions = []
    for path_images in tqdm(path_list):
        try:
            print(f'>>>> SCANNING FOLDER FOR IMAGES: {path_images}')
            # Get number of channels
            n_c_key = [n_c_key for n_c_key in n_c_keys if (n_c_key in str(path_images))]
            if len(n_c_key) == 0:
                n_c = n_c_default
                print(f'Will use default number of channels: {n_c}')
            elif len(n_c_key) == 1:
                n_c = n_c_dict[n_c_key[0]]
                print(f'Will use pre-defined channel number {n_c} for round {n_c_key[0]}')
            else:
                raise Exception(f'ERROR: multiple matches for n_c found, verify strings: {n_c_key}')
            # >>> Scan folder to get all files
            file_list = []
            for f_image in path_images.glob('*.tif'):
                # >> Only NDTIFF
                if ('NDTiffStack' in str(f_image.stem)):
                    file_list.append(str(f_image))
                else:
                    print('  Image name suggests that this is not a NDTiff image, will skip this one.')
                    continue
            file_list = sorted(file_list)
            n_files = len(file_list)
            print(f'   Found images: {file_list}')
            # >> Create path to save images
            round_name = path_images.name
            for old, new in string_replacements_path:
                round_name = re.sub(old, new, round_name, count=0, flags=0)
            path_save = folder_save / round_name
            print(f'  Images will be saved in folder: {path_save}')
            path_save.mkdir(parents=True, exist_ok=True)

            # >>>>>  Loop over images and split channels

            i_total = 0  # Total number of processed slices
            i_file = 0  # Index of currently loaded file

            # load first image
            print(f'     Loading image: {file_list[i_file]}')
            img = imread(file_list[i_file])
            print(f'     Image shape: {img.shape} {path_images.name}')
            #if img.shape[-1] == 3:
            #    img = img.transpose(2, 0, 1)  # Transpose to (z, x, y)
            #img = img.transpose(1, 0, 2, 3)
            #n_slices = img.shape[0]
            if img.ndim == 4:
                img = np.concatenate([img[i] for i in range(len(img))])
            if img.ndim == 5:
                to_concatenate = []
                for pos in range(n_pos):
                    for ch in range(n_c):
                        to_concatenate.append(img[pos,ch])
                img = np.concatenate(to_concatenate)

            print(f'     Image shape after reshape: {img.shape} {path_images.name}')

            n_slices = img.shape[0]
            assert n_slices == n_z * n_c * n_pos, "image dimension problem"
            print("CHECK WITH FLORIAN WHY THE IMAGE SHAPE ARE NOT THE SAME")


            for i_pos in tqdm(range(0, n_pos)):
                for i_c in range(0, n_c):

                    # Index of current image
                    i_img = (i_pos) * n_c + i_c

                    # Start and end index of current image
                    i_start = (i_img * n_z)
                    i_end = (i_start + n_z - 1)

                    # Correct for all slices loaded from previous images
                    i_start = i_start - i_total
                    i_end = i_end - i_total

                    print(f' start-end: {i_start}:{i_end}')

                    # >> Decide if next image should be loaded
                    if i_end <= n_slices:
                        img_save = img[i_start:i_end, :, :]
                    # Next image needs to be loaded
                    elif (i_end > n_slices):

                        # Extract remaining image
                        img_tmp_1 = img[i_start:n_slices, :, :]

                        # Load next image
                        i_file = i_file + 1
                        i_total = i_total + n_slices

                        if i_file >= n_files:
                            raise Exception(
                                'ERROR should open another file to continue, but there are not files left. \n check n_z and n_pos parameter')
                        print(f'     Loading image: {file_list[i_file]}')
                        img = imread(file_list[i_file])
                        print(f'     Image shape: {img.shape}')
                        if img.shape[-1] == 3:
                            img = img.transpose(2, 0, 1)
                        n_slices = img.shape[0]

                        # Load remainder of image
                        img_tmp_2 = img[0:i_end - i_total, :, :]

                        # Add images
                        img_save = np.concatenate((img_tmp_1, img_tmp_2), axis=0)

                    # File name
                    name_save = str(f_image.stem)
                    for old, new in string_replacements_file:
                        name_save = re.sub(old, new, name_save, count=0, flags=0)
                    name_save = name_save + f'_pos{i_pos}' + f'_ch{i_c}' + '.tif'

                    # Save image
                    f_save = path_save / name_save
                    print(f'  Image will be saved as {str(f_save)}')
                    print(f' SAVED    Image shape: {img_save.shape}')
                    imsave(str(f_save), img_save, check_contrast=False)
        except Exception as e:
            print(f'ERROR: {e}')
            list_exceptions.append((e, path_images.name))
    return list_exceptions
