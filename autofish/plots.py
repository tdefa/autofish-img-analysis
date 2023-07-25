

import tifffile
import numpy as np

import napari

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

