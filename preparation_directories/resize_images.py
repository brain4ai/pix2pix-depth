import cv2
import os


def resize_images(path, path_out, width, height, inter=cv2.INTER_AREA):
    '''
    -Este script permite redimensionar las imagenes de un directorio completo

    :param path: Path donde se encuentran las imagenes
    :param path_out: Path donde se exportarán las imagenes
    :param width: Ancho en pixeles de la imagen que deseamos
    :param height: Alto en pixeles de la imagen que deseamos
    :param inter: Metodo de interpolacion que usara opencv para el resize de la imagen
    :return:
    '''

    #Obtain files names
    files = os.listdir(path)
    print("[INFO] Number of files detected : " + str(len(files)))

    for f in files:
        # Complete path
        path_file = path + f

        # Read image
        im = cv2.imread(path_file)

        # Resize image
        res = cv2.resize(im, (width, height), interpolation=inter)

        # Save image
        try:
            path_save = path_out + f
            cv2.imwrite(path_save, res)
            print("[INFO] Image saved at : " + path_save)
        except:
            print("[ERROR] : Save Image Error")
            break

# USO
p1 = "../dataset/dataset/concatenate-all/"
p2 = p1
width = 512
height = 256
resize_images(p1, p2, width, height) #el parametro inter es opcional