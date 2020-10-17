import os
import numpy as np
import cv2


def stack_images(dir1, dir2, dir3, extension=".png"):
    '''
    -Este script permite unir en una misma imagen, dos imágenes de diferentes path
    quedando algo así como modo panorámica
    -Coge las imagenes del directorio 1 y las une con las del directorio 2 y las exporta
    a un directorio 3

    :param dir1: Path donde se encuentran las imagenes del directorio 1
    :param dir2: Path donde se encuentran las imagenes del directorio 2
    :param dir3: Path donde se guardaremos las imagenes apiladas
    :param extension: Formato al cual exportaremos la imagen apilada finalmente
    :return:
    '''

    # Compare the number of images on both directories, must be the same
    files_dir1 = os.listdir(dir1)
    print("[INFO] Number of files on dir1 : " + str(len(files_dir1)))
    print(files_dir1)
    files_dir2 = os.listdir(dir2)
    print("[INFO] Number of files on dir2 : " + str(len(files_dir2)))

    if len(files_dir1) == len(files_dir2):
        # Iterate over images on both directories
        for counter, f1 in enumerate(files_dir1):
            # Extract extensions
            file1, ext1 = os.path.splitext(f1)
            print("[INFO] Extension detected for file1: " + ext1)
            file2, ext2 = os.path.splitext(files_dir2[counter])
            print("[INFO] Extension detected for file2: " + ext2)

            # Construct the path to read images
            file1_path = dir1 + file1 + ext1
            file2_path = dir2 + file2 + ext2

            # Open images and concatenate
            img1 = cv2.imread(file1_path)
            img2 = cv2.imread(file2_path)
            vis = np.concatenate((img1, img2), axis=1)

            # Save image
            try:
                save_path = dir3 + file1 + extension
                cv2.imwrite(save_path, vis)
                print("[INFO] Image Saved on directory : " + save_path)
            except:
                print("[ERROR] : Save Image Error")
                break
    else:
        print("[ERROR] There are not the same number of images to concatenate on the directories")

# USO
p1 = "../dataset/images-all/"
p2 = "../dataset/depth-all/"
p3 = "../dataset/dataset/concatenate-all/"
p4 = ".png"
stack_images(p1, p2, p3, p4) # Como puede verse, p4 es opcional, por defecto exporta a .png