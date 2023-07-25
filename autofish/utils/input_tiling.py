


import tifffile
import numpy as np
from pathlib import Path
from tqdm import tqdm





def tilling_input(image,
        nb_tile_x = 3,
    nb_tile_y = 3,
    path_to_save = "/media/tom/T7/sp_data/In_situ_Sequencing_16/dapi_tile/" ,
    save_mask = True):


    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    if image.ndim == 3:
        raise NotImplemented("image should be 2D, not implemented for 3D yet")

    tile_y = image.shape[0] // nb_tile_y
    tile_x = image.shape[1] // nb_tile_x
    print(tile_y, tile_x)
    dico_stich = {}
    for x in tqdm(range(nb_tile_x)):
        for y in range(nb_tile_y):
            tile = image[y*tile_y: np.min([(y+1)*tile_y + 200, image.shape[0]]),

                   x*tile_x: np.min([(x+1)*tile_x + 200, image.shape[1]])]
            print(f" tile shape {tile.shape}")

            image_name =  "tile_y_{}_x_{}.tif".format(y, x)
            dico_stich[image_name] = [y*tile_y,  x*tile_x]
            if save_mask:
                tifffile.imwrite(path_to_save + image_name, tile)

    return dico_stich








if __name__ == '__main__':




    image = tifffile.imread("/media/tom/T7/sp_data/In_situ_Sequencing_16/dapi/Base_1_stitched-1.tif")
    dico_stich = tilling_input(image,
                  nb_tile_x=10,
                  nb_tile_y=10,
                  path_to_save="/media/tom/T7/sp_data/In_situ_Sequencing_16/dapi_tile_10_10/",
                               save_mask=False)
    np.save("/media/tom/T7/sp_data/In_situ_Sequencing_16/dapi_tile/dico_stich_10_10.npy", dico_stich)




    # draf Tilling an image
