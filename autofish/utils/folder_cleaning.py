

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm














if __name__ ==  "__main__":


    path_main_folder = "/media/tom/Transcend/lustr2023/images"
    path_folder_of_folder_mip = "/media/tom/Transcend/lustr2023/images/image_mip"

    for path_folder_to_mip in tqdm(list(Path(path_main_folder).glob("r*"))):

        path_folder_to_mip = str(path_folder_to_mip)

        #path_folder_to_mip =   "/media/tom/Transcend/lustr2023/images/r1_Cy3"


        name_round = Path(path_folder_to_mip).name
        Path(path_folder_of_folder_mip +"/" + name_round ).mkdir(exist_ok=True)

        print(path_folder_of_folder_mip +"/" + name_round)




        for image in tqdm(list(Path(path_folder_to_mip).glob("*.tif"))):
            img = tifffile.imread(image)
            img_mip = np.max(img, axis=0)
            tifffile.imsave(str(path_folder_of_folder_mip) + "/" + name_round + "/" + image.name, img_mip)