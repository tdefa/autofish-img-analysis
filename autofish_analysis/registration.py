
# %%
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .spots_detection import remove_double_detection


def compute_euler_transform(fixed_image,
                            moving_image,  # works ok
                            ndim = 2,
                            numberOfHistogramBins=500,
                            sampling_percentage=0.05,
                            max_rotation_accepted = 0.02,
                            ########## ITK parameters, leave as default unless not converging  ##########
                            learningRate=1,
                            numberOfIterations=100,
                            convergenceMinimumValue=1e-6,
                            convergenceWindowSize=10,
                            shrinkFactors = [4 ,2 ,1],
                            smoothingSigmas=[2 ,1 ,0],):

    """
    :param fixed_image: image to register to
    :param moving_image: image to register
    :param ndim: 2 or 3
    :return: thetha, x_translation, y_translation
    fixed_image + (x_translation, y_translation) = moving_image
    """

    assert ndim in [2, 3]
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)

    if ndim == 3:
        raise NotImplementedError("3d registration not implemented yet")
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

    else:

        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                              moving_image,
                                                              sitk.Euler2DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=numberOfHistogramBins)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
    learningRate=learningRate,
    numberOfIterations=numberOfIterations,
    convergenceMinimumValue=convergenceMinimumValue,
    convergenceWindowSize=convergenceWindowSize,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = shrinkFactors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothingSigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print(final_transform.GetParameters())

    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print(
    "Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()
    )
    )
    final_metric_value = registration_method.GetMetricValue()
    if ndim == 2:
        thetha = final_transform.GetParameters()[0]
        x_translation = final_transform.GetParameters()[1] ## itk do not uxe python convention
        y_translation = final_transform.GetParameters()[2]
        assert thetha < max_rotation_accepted,  f"the image is rotated more than the max_rotation_accepted :  {max_rotation_accepted} radians"
        return final_metric_value, thetha, x_translation, y_translation
    elif ndim == 3:
        raise NotImplementedError("3d registration not implemented yet")


from skimage.transform import AffineTransform, warp


def folder_translation(folder_of_rounds = "/media/tom/T7/Stitch/acquisition/",  # works ok
                       fixed_round_name = "r1_bc1",
                       folder_regex = 'r',
                       chanel_regex = 'ch1',
                       registration_repeat = 5,
                       position_naming = True,
                       sigma_gaussian_filter = 0.8):
    """

    :param folder_of_rounds:  folder containing the rounds
    :param fixed_round_name:   name of the round to register to
    :param folder_regex:   regex to find the rounds
    :param chanel_regex: regex to find the channels
    :param registration_repeat: number of time the registration is initialized
    :param position_naming:
    :param sigma_gaussian_filter: sigma of the gaussian filter applied to the image before registration
    :return:
    dict of dict of dict such that
     dico_translation[position][fixed_round_name][moving round] {'thetha': rotation in radians,
                                                                'x_translation': x_translation,
                                                                'y_translation': y_translation}
    """

    #### get image names form static folder

    dico_translation = {}
    list_path_img_static = list(Path(folder_of_rounds).joinpath(fixed_round_name).glob(f'{chanel_regex}'))
    for path_img_static in tqdm(list_path_img_static):
        print()
        print(path_img_static.name)
        position_img_static = "pos" + path_img_static.name.split('pos')[1].split('_')[0]
        dico_translation[position_img_static] = {}
        dico_translation[position_img_static][fixed_round_name] = {}
        fixed_image = np.amax(tifffile.imread(path_img_static), 0).astype(float)
        for path_round in list(Path(folder_of_rounds).glob(f'{folder_regex}')):
            if path_round.name == fixed_round_name:
                continue
            try:
                print(path_round.name)

                if position_naming:
                        moving_image_name = path_round.name + '_p' + path_img_static.name.split('_p')[1]
                else:
                    moving_image_name = path_img_static.name
                moving_image = np.amax(tifffile.imread(path_round.joinpath(moving_image_name)), 0).astype(float)
                assert fixed_image.shape == moving_image.shape
                assert fixed_image.ndim == 2
                rep_list = []
                if sigma_gaussian_filter is not None:
                    fixed_image = gaussian_filter(fixed_image, sigma=sigma_gaussian_filter)
                    moving_image = gaussian_filter(moving_image, sigma=sigma_gaussian_filter)
                try:
                    for rep in range(registration_repeat):

                        final_metric_value, thetha, x_translation, y_translation = compute_euler_transform(
                                            fixed_image = fixed_image,
                                                moving_image = moving_image,  # works ok
                                                ndim = fixed_image.ndim)
                        rep_list.append([final_metric_value, thetha, x_translation, y_translation])
                except RuntimeError:
                    print(f"registration failed {(path_img_static.name, fixed_round_name)}")
                    rep_list.append([100, 0, 0, 0])
                    continue
            except FileNotFoundError as e:
                    print(e)
                    print(f"no image found for {path_round.name}")
                    continue

            min_index = np.argmin(np.array(rep_list)[:, 0])
            thetha, x_translation, y_translation = rep_list[min_index][1:]
            dico_translation[position_img_static][fixed_round_name][path_round.name] = {'thetha': thetha,
                                                                                          'x_translation': x_translation,
                                                                                            'y_translation': y_translation}
    return dico_translation




def spots_registration(dict_spots,
                    dict_translation,
                     fixed_round_name = "r1_bc1",
                     check_removal=False,
                     threshold_merge_limit=None,
                     scale_xy=0.103,
                     scale_z=0.300,
                     ):
    """
    register the spots to the reference round
    :param dict_spots:  dict of dict of dict containing the spots of all round and all images
    dict_spots[round][image][spot] = [x,y,z]
    :param dict_translation:  dict of dict of dict containing the translation of all round and all images
    dict_translation[position][fixed_round_name][moving round] {'thetha': rotation in radians,
                                                                'x_translation': x_translation,
                                                                'y_translation': y_translation}
    :param fixed_round_name:  name of the round to register to
    :param check_removal:   (leave it False use only for test)
    :param threshold_merge_limit: (leave it None use only for test)
    :param scale_xy: pixel size in xy
    :param scale_z: pixel size in z
    :return:
    dict_spots_registered_df : dict of dict of dict containing the spots of all round and all images registered to the reference round in a dataframe format
    dict_spots_registered : dict of dict of dict containing the spots of all round and all images registered to the reference round in array format
    missing_data : list of missing data if any
    """

    if threshold_merge_limit is None:
        threshold_merge_limit = 3 * scale_xy
    ### register each round to the ref round
    dict_spots_registered = {}
    missing_data = []
    image_list = list(dict_spots[fixed_round_name].keys())
    for round_t in tqdm(list(dict_spots.keys())):
        dict_spots_registered[round_t] = {}
        for image_name_fixed_round_name in tqdm(image_list):
            ### determine the position of the image , the number after "pos" is the position
            image_position = "pos" + image_name_fixed_round_name.split('pos')[1].split('_')[0]
            image_name = None
            for k in dict_spots[round_t].keys():
                if image_position == "pos" + k.split('pos')[1].split('_')[0]:
                    image_name = k
                    break
            # print(image_name)
            if image_name not in dict_spots[round_t].keys():
                dict_spots_registered[round_t][image_position] = []
                missing_data.append([round_t, image_name])
                print(f'missing data {round_t} {image_name}')

                continue

            if round_t not in dict_translation[image_position][fixed_round_name].keys() and round_t != fixed_round_name:
                missing_data.append([round_t, image_name])
                dict_spots_registered[round_t][image_position] = []
                print(f'missing data {round_t} {image_name}')
                continue
            if round_t == fixed_round_name:
                x_translation = 0
                y_translation = 0
            else:
                x_translation = dict_translation[image_position][fixed_round_name][round_t]['x_translation']
                y_translation = dict_translation[image_position][fixed_round_name][round_t]['y_translation']

            if image_name in dict_spots[round_t]:
                dict_spots_registered[round_t][image_position] = dict_spots[round_t][image_name] - np.array \
                    ([0, y_translation, x_translation])
            else:
                dict_spots_registered[round_t][image_position] = dict_spots[round_t][image_position] - np.array \
                    ([0, y_translation, x_translation])
            if check_removal:
                clean_spots = remove_double_detection(input_array = dict_spots_registered[round_t][image_position],
                    threshold =threshold_merge_limit,
                    scale_z_xy = np.array([scale_z, scale_xy, scale_xy]))
                if len(clean_spots) != len(dict_spots_registered[round_t][image_position]):
                    print("double detection removal bug during the detection of spots")
                    dict_spots_registered[round_t][image_position] = clean_spots
    ## generate a dataframe
    dict_spots_registered_df = {}
    list_image_positon = list(dict_spots_registered[fixed_round_name].keys())

    for image_position in list_image_positon:
        x_list = []
        y_list = []
        z_list = []
        round_list = []
        for round in dict_spots_registered:
            try:
                spots_list = dict_spots_registered[round][image_position]
                x_list += list(spots_list[:, 2])
                y_list += list(spots_list[:, 1])
                z_list += list(spots_list[:, 0])
                round_list += [round] * len(spots_list)
            except KeyError:
                print(f"missing data {round} {image_position}")
                continue
        dict_spots_registered_df[image_position] = pd.DataFrame({'x': x_list,
                                                                 'y': y_list,
                                                                 'z': z_list,
                                                                 'round': round_list})
    return dict_spots_registered_df, dict_spots_registered, missing_data




def spots_assignement(dict_spots_registered_df,
                      path_to_masks = "/media/tom/Transcend/autofish_test/r1/segmentation_mask",
                      files_mask_extension = "*.tif*" ,
                      in_place = False,):

    if not in_place:
        new_dict_spots_registered_df = {}
    path_to_masks = Path(path_to_masks)
    for ind_path_mask in tqdm(list(path_to_masks.glob(f"{files_mask_extension}"))):
        print(ind_path_mask)
        if 'tif' in files_mask_extension:
            mask = tifffile.imread(ind_path_mask)
        elif 'npy' in files_mask_extension:
            mask = np.load(ind_path_mask)
        else:
            raise ValueError("mask extension not supported")
        position = "pos" + ind_path_mask.name.split('pos')[1].split('_')[0].split('.')[0]
        z_range = range(mask.shape[0])
        y_range = range(mask.shape[1])
        x_range = range(mask.shape[2])
        if position not in dict_spots_registered_df.keys():
                raise ValueError(f"no spots registered for {position}")
        df_spot = dict_spots_registered_df[position]
        list_cell_assignment = []
        for index, spots in df_spot.iterrows():
            z, y, x = round(spots.z), round(spots.y), round(spots.x)
            if z not in z_range or y not in y_range or x not in x_range:
                list_cell_assignment.append(-1)
            else:
                list_cell_assignment.append(mask[z, y, x])
        df_spot['cell_assignment'] = list_cell_assignment
        if not in_place:
            new_dict_spots_registered_df[position] = df_spot
        else:
            dict_spots_registered_df[position] = df_spot

    if not in_place:
        return new_dict_spots_registered_df
    else:
        return dict_spots_registered_df








###################################
## plotting function to check registration
####################################

def shift(image, translation):
    assert image.ndim == 2
    transform = AffineTransform(translation=translation)
    shifted = warp(image, transform, mode='constant', preserve_range=True)

    shifted = shifted.astype(image.dtype)
    return shifted


def plot_registrered_image(

    dico_translation,
        path_image1="/media/tom/Transcend/lustr2023/images/r1_Cy3/r1_pos22_ch0.tif",
        path_image2="/media/tom/Transcend/lustr2023/images/r11/r11_pos22_ch0.tif",
    plot_napari = True,):

    image1 =np.amax(tifffile.imread(path_image1), 0)
    image2 = np.amax(tifffile.imread(path_image2), 0)
    pos1 = 'pos' + Path(path_image1).name.split('pos')[1].split('_')[0]
    pos2 = 'pos' + Path(path_image2).name.split('pos')[1].split('_')[0]
    print(pos1, pos2)
    assert pos1 == pos2
    round1 = "r" + Path(path_image1).name.split('r')[1].split('_')[0]
    round2 = "r" + Path(path_image2).name.split('r')[1].split('_')[0]

    x_translation = dico_translation[pos1][round1][round2]['x_translation']
    y_translation = dico_translation[pos1][round1][round2]['y_translation']

    shifted_image = shift(image2, translation=(x_translation, y_translation))

    from matplotlib import pyplot as plt

    ### plot shifted image and image 1
    if plot_napari:
        import napari
        viewer = napari.view_image(image1, name="image1")
        viewer.add_image(image2, name="image2")
        viewer.add_image(shifted_image, name="shifted_image2")

    else:
        fig, ax = plt.subplots(figsize =  (15,15),ncols=2,)
        ax[0].imshow(image1, alpha=0.5, cmap= "gray")
        ax[0].imshow(image2, alpha=0.5)
        ax[1].imshow(image1, alpha=0.5)
        ax[1].imshow(shifted_image)
     #   ax[1].imshow(shifted_image)

        plt.show()

