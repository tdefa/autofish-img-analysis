

#%%


import re
from pathlib import Path

import bigfish.stack as stack
import numpy as np
import tifffile
from bigfish.detection.utils import (get_object_radius_pixel, get_spot_surface,
                                     get_spot_volume)
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.exposure import rescale_intensity
from tqdm import tqdm

############### redfine compute_snr_spots #################

def compute_snr_spots(image, spots, voxel_size, spot_radius, return_list = False): ## founction custom from bigfish
    """Compute signal-to-noise ratio (SNR) based on spot coordinates.

    .. math::

        \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
        \\mbox{mean(background)}}{\\mbox{std(background)}}

    Background is a region twice larger surrounding the spot region. Only the
    y and x dimensions are taking into account to compute the SNR.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray
        Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
        One coordinate per dimension (zyx or yx coordinates).
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.

    Returns
    -------
    snr : float
        Median signal-to-noise ratio computed for every spots.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_range_value(image, min_=0)
    stack.check_array(
        spots,
        ndim=2,
        dtype=[np.float32, np.float64, np.int32, np.int64])
    stack.check_parameter(
        voxel_size=(int, float, tuple, list),
        spot_radius=(int, float, tuple, list))

    # check consistency between parameters
    ndim = image.ndim
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but spots are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(spot_radius, (tuple, list)):
        if len(spot_radius) != ndim:
            raise ValueError(
                "'spot_radius' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        spot_radius = (spot_radius,) * ndim

    # cast spots coordinates if needed
    if spots.dtype == np.float64:
        spots = np.round(spots).astype(np.int64)

    # cast image if needed
    image_to_process = image.copy().astype(np.float64)

    # clip coordinate if needed
    if ndim == 3:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
        spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
    else:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

    # compute radius used to crop spot image
    radius_pixel = get_object_radius_pixel(
        voxel_size_nm=voxel_size,
        object_radius_nm=spot_radius,
        ndim=ndim)
    radius_signal_ = [np.sqrt(ndim) * r for r in radius_pixel]
    radius_signal_ = tuple(radius_signal_)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * 2 for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(np.int)
    radius_background = np.ceil(radius_background_).astype(np.int)

    # loop over spots
    snr_spots = []
    background_spots = []
    max_signal_spots = []
    for spot in spots:

        # extract spot images
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx
        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            spot_background_, _ = get_spot_volume(
                image_to_process, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border (along y and x dimensions)
            expected_size = (2 * radius_background_yx + 1) ** 2
            actual_size = spot_background.shape[1] * spot_background.shape[2]
            if expected_size != actual_size:
                continue

            # remove signal from background crop
            spot_background[:,
                            edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border
            expected_size = (2 * radius_background_yx + 1) ** 2
            if expected_size != spot_background.size:
                continue

            # remove signal from background crop
            spot_background[edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        # compute mean background
        mean_background = np.mean(spot_background)

        # compute standard deviation background
        std_background = np.std(spot_background)

        # compute SNR
        snr = (max_signal - mean_background) / std_background
        snr_spots.append(snr)
        background_spots.append(mean_background)
        max_signal_spots.append(max_signal)

    if return_list:
        return snr_spots, background_spots, max_signal_spots

    #  average SNR
    if len(snr_spots) == 0:
        snr = 0.
    else:
        snr = np.median(snr_spots)
    return snr






def compute_snr_for_each_spots(spots,
                               im_fish,
                               voxel_size = [270, 108, 108],
                                        spot_radius=300,):
    spots = np.array(spots)
    snr = compute_snr_spots(im_fish, spots,
                     voxel_size=voxel_size,
                     spot_radius = spot_radius)
    return snr



def mean_cos_tetha(gz, gy, gx, z, yc, xc, order = 3):



    """
    Args:
        gy (): gz, gy, gx = np.gradient(rna_gaus)
        gx ():
        z ():  z coordianate of the detected spots
        yc (): yc coordianate of the detected spots
        xc (): xc coordianate of the detected spots
        order (): number of pixel in xy away from the detected spot to take into account
    Returns:
    """

    import math
    list_cos_tetha = []
    for i in range(xc-order, xc+order+1):
        for j in range(yc-order, yc+order+1):
            if i - xc < 1 and j-yc < 1:
                continue
            if i < 0 or i > gx.shape[2]-1:
                continue
            if j < 0 or j > gx.shape[1]-1:
                continue
            vx = (xc - i)
            vy = (yc - j)

            cos_tetha = (gx[z, j, i]*vx + gy[z, j, i]*vy) / (np.sqrt(vx**2 +vy**2) * np.sqrt(gx[z, j, i]**2 + gy[z, j, i]**2) )
            if math.isnan(cos_tetha):
                continue
            list_cos_tetha.append(cos_tetha)
    return np.mean(list_cos_tetha)




def compute_symetry_coef_for_each_spots(spots,
                                        im_fish,
                                        sigma = 1.3,
                                        order=5):


    rna_gaus = ndimage.gaussian_filter(im_fish, sigma)
    gz, gy, gx = np.gradient(rna_gaus)
    list_symmetry_coef = []
    for s in spots:
        symmetry_coef  = mean_cos_tetha(gz, gy, gx,
                   z=s[0], yc=s[1],
                   xc=s[2], order=order)
        list_symmetry_coef.append(symmetry_coef)
    return list_symmetry_coef


def compute_intensity_list(spots,
                               im_fish,):
    intensity_list = []
    for s in spots:
        intensity = im_fish[int(round(s[0])),
                            int(round(s[1])),
                            int(round(s[2]))]
        intensity_list.append(intensity)
    return intensity_list



def compute_quality_all_rounds(
                            dict_spots,
                            folder_of_rounds = "/media/tom/T7/Stitch/acquisition/",
                            round_name_regex="r",
                            image_name_regex="opool1_1_MMStack_3",
                            channel_name_regex="ch1",
                            file_extension = ".ti",
                            voxel_size=[300, 108, 108],
                            spot_radius=300,
                                sigma=1.3,
                                order=5,
                            compute_sym = True,
                            return_list = True,
                             ):

    dico_signal_quality = {}
    for path_round in tqdm(list(Path(folder_of_rounds).glob(f"{round_name_regex}"))):
        print()
        print(path_round.name)
        dico_signal_quality[path_round.name] = {}
        for path_rna_img in tqdm(list(path_round.glob(f"{channel_name_regex}{file_extension}*"))):

            if not re.match(image_name_regex, path_rna_img.name) :   #image_name_regex not in path_rna_img.name:

                continue
            print(path_rna_img.name)

            pos = "pos" + path_rna_img.name.split("pos")[1].split("_")[0]
            rna_img = tifffile.imread(path_rna_img)
            if pos in dict_spots[path_round.name].keys():
                spots = dict_spots[path_round.name][pos]
            else:
                spots = dict_spots[path_round.name][path_rna_img.name]

            intensity_list = compute_intensity_list(spots, rna_img,)
            if return_list:
                snr_spots, background_spots, max_signal_spots = compute_snr_spots(rna_img, spots,
                         voxel_size=voxel_size,
                         spot_radius = spot_radius,
                        return_list = return_list)
            else:
                snr_spots = compute_snr_spots(rna_img, spots,
                         voxel_size=voxel_size,
                         spot_radius = spot_radius,
                        return_list = return_list)
                background_spots = None
                max_signal_spots = None
            if compute_sym:
                symmetry_coef_list = compute_symetry_coef_for_each_spots(spots,
                                                                         rna_img,
                                                                         sigma=sigma,
                                                                         order=order
                                                                         )
            else:
                symmetry_coef_list = None
            dico_signal_quality[path_round.name][pos] = {"intensity": intensity_list,
                                                                        "snr": snr_spots,
                                                                        "symmetry_coef": symmetry_coef_list,
                                                                       "background": background_spots,
                                                                       "max_signal": max_signal_spots,}
    return dico_signal_quality

def folder_signal_quality(
                                dict_spots,
                                folder_of_rounds,
                                round_name_regex,
                                image_name_regex,
                                channel_name_regex,
                                file_extension,
                                spot_radius,
                                sigma = 1.3,
                                order=5,
                                compute_sym=False,
                                ):

    dico_signal_quality = compute_quality_all_rounds(
                                dict_spots,
                                folder_of_rounds = folder_of_rounds,
                                round_name_regex=round_name_regex,
                                image_name_regex=image_name_regex,
                                channel_name_regex=channel_name_regex,
                                file_extension = file_extension,
                                spot_radius=spot_radius,
                                    sigma=sigma,
                                    order=order,
                                compute_sym = compute_sym,
                                return_list = True,
                                 )

    median_intensity = []
    median_snr = []
    median_symmetry_coef = []
    median_background = []
    for round_str in dico_signal_quality:
        mean_intensity = []
        mean_snr = []
        mean_symmetry_coef = []
        mean_background = []

        for image in dico_signal_quality[round_str]:
            mean_intensity += dico_signal_quality[round_str][image]["intensity"]
            mean_background += dico_signal_quality[round_str][image]["background"]
            mean_snr += dico_signal_quality[round_str][image]["snr"]
            if compute_sym:
                mean_symmetry_coef += dico_signal_quality[round_str][image]["symmetry_coef"]

        median_intensity.append(np.median(mean_intensity))
        median_background.append(np.median(mean_background))
        median_snr.append(np.median(mean_snr))
        if compute_sym:
            median_symmetry_coef.append(np.median(mean_symmetry_coef))


    if compute_sym:
        df = pd.DataFrame({"round":list(dico_signal_quality.keys()),
                          "median_intensity":median_intensity,
                          "median_background":median_background,
                          "median_snr":median_snr,
                          "median_symmetry_coef":median_symmetry_coef})
    else:
        df = pd.DataFrame({"round":list(dico_signal_quality.keys()),
                          "median_intensity":median_intensity,
                          "median_background":median_background,
                          "median_snr":median_snr})
    df.to_csv(Path(folder_of_rounds) / "signal_quality.csv", index=False)
    return df

def plot_spots_folder(dico_spots,
                    round_folder_path = "/media/tom/T7/Stitch/acquisition/",
                    round_name_regex="r",
                    image_name_regex="opool1_1_MMStack_3",
                    channel_name_regex="ch1",
                    file_extension = ".ti",
                      spot_size = 5,
                      figsize=(10, 10),
                      folder_save= '/media/tom/Transcend/autofish_test/figure'):


    print(folder_save)

    folder_save = Path(folder_save)
    print(folder_save)
    folder_save.mkdir(exist_ok=True, parents=True)
    for path_round in tqdm(list(Path(round_folder_path).glob(f"{round_name_regex}"))[4:] + list(Path(round_folder_path).glob(f"{round_name_regex}"))):
        print()
        print(path_round.name)
        for path_rna_img in tqdm(list(path_round.glob(f"{channel_name_regex}{file_extension}*"))):
            print(path_rna_img.name)
            if not re.match(image_name_regex, path_rna_img.name) :   #image_name_regex not in path_rna_img.name:
                continue
            print(path_rna_img.name)
            rna_img = tifffile.imread(path_rna_img)
            spots = dico_spots[path_round.name][path_rna_img.name]
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            rna_img = np.amax(rna_img, axis=0)
            pa_ch1, pb_ch1 = np.percentile(rna_img, (1, 99))
            input = rescale_intensity(rna_img, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
            ax[0].imshow(input, cmap="gray")
            ax[0].scatter(spots[:, 2], spots[:, 1], s=spot_size, c="red")
            ax[1].imshow(input, cmap="gray")
            plt.show()
            fig.savefig(folder_save / f"{path_rna_img.name}_spots.png")






