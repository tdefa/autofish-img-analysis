



import numpy as np
#from bigfish.detection import  compute_snr_spots
from bigfish_custom.detection import compute_snr_spots

from scipy import ndimage

from pathlib import Path
import tifffile
from tqdm import tqdm
import re
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity


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
    todo add checking
    todo extend to 3D checking
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
                            round_folder_path = "/media/tom/T7/Stitch/acquisition/",
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

    from bigfish_custom.detection import compute_snr_spots
    dico_signal_quality = {}
    for path_round in tqdm(list(Path(round_folder_path).glob(f"{round_name_regex}"))):
        print()
        print(path_round.name)
        dico_signal_quality[path_round.name] = {}
        for path_rna_img in tqdm(list(path_round.glob(f"{channel_name_regex}{file_extension}*"))):

            if image_name_regex not in path_rna_img.name:
                continue
            print(path_rna_img.name)

            rna_img = tifffile.imread(path_rna_img)
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
            dico_signal_quality[path_round.name][path_rna_img.name] = {"intensity": intensity_list,
                                                                        "snr": snr_spots,
                                                                        "symmetry_coef": symmetry_coef_list,
                                                                       "background": background_spots,
                                                                       "max_signal": max_signal_spots,}
    return dico_signal_quality








def plot_spots_folder(dico_spots,
                    round_folder_path = "/media/tom/T7/Stitch/acquisition/",
                    round_name_regex="r",
                    image_name_regex="opool1_1_MMStack_3",
                    channel_name_regex="ch1",
                    file_extension = ".ti",
                      spot_size = 5,
                      figsize=(10, 10),
                      folder_save= '/media/tom/Transcend/autofish_test/figure'):
    import re

    folder_save = Path(folder_save)
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






