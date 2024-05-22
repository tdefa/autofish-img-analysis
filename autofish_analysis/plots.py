

import napari
import numpy as np
import tifffile
from skimage.exposure import rescale_intensity
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

path_project = "/home/tom/Bureau/2023-10-06_LUSTRA/"

dict_spots = np.load('/home/tom/Bureau/2023-10-06_LUSTRA/dict_spots_polaris.npy',
                     allow_pickle=True).item()

path_save = Path("/home/tom/Bureau/2023-10-06_LUSTRA/detection_7janv_polris/")
def plot_detected_spots(dict_spots,
                        path_project = "/home/tom/Bureau/2023-10-06_LUSTRA/",
                        figsize = (20,20),
                        color = "red",
                        spots_size = 1,
                        rescal_percentage = 99.8,
                        path_save = None,
                        ):

    for round in tqdm(dict_spots):
        if path_save is None:
            path_save = Path(path_project) / (round +"/"+ "spots_detection_fi")
        path_save.mkdir(exist_ok=True, parents=True)

        for image in tqdm(dict_spots[round]):
            img = tifffile.imread(Path(path_project) / (round + "/" +  image) )
            spots_array = dict_spots[round][image]
            if img.ndim == 3:
                img = np.amax(img, 0)
            else:
                assert img == 2


            fig, ax = plt.subplots(figsize =  figsize,ncols=2, nrows=2)
            ax[0, 0].imshow(img)
            ax[1, 0].imshow(img)

            pa_ch1, pb_ch1 = np.percentile(img, (1, rescal_percentage))
            img_rescale = rescale_intensity(img, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)
            ax[0, 1].imshow(img_rescale)
            ax[1, 1].imshow(img_rescale)
            if spots_array.shape[-1] == 0:
                continue

            if spots_array.ndim == 3 and spots_array.shape[-1] == 2:
                spots_array = spots_array[0]
                if len(spots_array) == 0:
                    continue
                ax[1, 1].scatter(spots_array[:, 1], spots_array[:, 0],
                                 c=color,
                                 s=spots_size)
            else:
                ax[1, 1].scatter(spots_array[:, 2], spots_array[:, 1],
                                 c=color,
                                 s=spots_size)

            fig.savefig(path_save / Path(image).stem )

            ### napaari
            img = tifffile.imread(Path(path_project) / (round + "/" +  image) )




            viewer = napari.viewer.Viewer()

            viewer.add_image(img, name='rna')
            # viewer.add_image(img_dapi, name='rna')

            viewer.add_points(spots_array, name='spots',
                              face_color='red', edge_color='red', size=5)
            break



def plot_registrered_image(
        dico_translation,
        path_image1="/media/tom/Transcend/lustr2023/images/r1_Cy3/r1_pos22_ch0.tif",
        path_image2="/media/tom/Transcend/lustr2023/images/r11/r11_pos22_ch0.tif",
        plot_napari = True,
        figsize = (10, 10)):

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

    print(f'x_translation {x_translation} y_translation {y_translation} ')

    shifted_image = shift(image2, translation=(x_translation, y_translation))

    ### plot shifted image and image 1
    if plot_napari:
        import napari
        viewer = napari.view_image(image1, name="image1")
        viewer.add_image(image2, name="image2")
        viewer.add_image(shifted_image, name="shifted_image2")

    else:

        pa_ch1, pb_ch1 = np.percentile(image1, (1, 99))
        image1 = rescale_intensity(image1, in_range=(pa_ch1, pb_ch1), out_range=np.uint8)

        pa_ch2, pb_ch2 = np.percentile(image2, (1, 99))
        image2 = rescale_intensity(image2, in_range=(pa_ch2, pb_ch2), out_range=np.uint8)

        pa_ch3, pb_ch3 = np.percentile(shifted_image, (1, 99))
        shifted_image = rescale_intensity(shifted_image, in_range=(pa_ch3, pb_ch3), out_range=np.uint8)


        fig, ax = plt.subplots(figsize =  figsize,ncols=2,)
        ax[0].imshow(image1, alpha=0.5, cmap= "RdGy_r")
        ax[0].imshow(image2, alpha=0.5, cmap= "Greens_r")
        ax[1].imshow(image1, alpha=0.5, cmap= "RdGy_r")
        ax[1].imshow(shifted_image, alpha=0.5,cmap= "Greens_r")

if __name__ == "__main__":






    img_rna = tifffile.imread("/media/tom/Transcend/lustr2023/images/r2/r2_pos0_ch0.tif")
    #img_dapi = tifffile.imread("/media/tom/T7/Stitch/acquisition/r1_bc1/opool1_1_MMStack_3-Pos_1_ch3.tif")
    #img_rna[53] = img_rna[52] / 2 + img_rna[54] / 2
    #img_rna = tifffile.imsave("/media/tom/T7/Stitch/acquisition/r6_bc7/opool1_1_MMStack_3-Pos_1_ch1.tif", img_rna)
    dico_spots = np.load('/media/tom/Transcend/lustr2023/images/23mai_dico_spots_local_detection0.npy', allow_pickle=True).item()


    viewer =  napari.viewer.Viewer()

    viewer.add_image(img_rna, name='rna')
    #viewer.add_image(img_dapi, name='rna')

    viewer.add_points(dico_spots['r2']['r2_pos0_ch0.tif'], name='spots',
                      face_color='red', edge_color='red', size=5)












    img_rna = tifffile.imread("/media/tom/T7/For_Thomas/For_Thomas/230506_Lamp3_clearing_17h_Anch_1PK_FLASH/Cell_01_w21 CY3.TIF")
    dico_spots = np.load('/media/tom/T7/For_Thomas/For_Thomas/10mai_dico_spots_local_detection0.npy',
                         allow_pickle=True).item()
    viewer =  napari.viewer.Viewer()
    viewer.add_image(img_rna, name='rna')
    #viewer.add_image(img_dapi, name='rna')

    viewer.add_points(dico_spots['230506_Lamp3_clearing_17h_Anch_1PK_FLASH']['Cell_01_w21 CY3.TIF'], name='spots',
                      face_color='red', edge_color='red', size=5)

