import os


def move_files_by_string(path_files, part_name_to_find, path_to_move):
    '''
    Mueve ficheros o imagenes a partir de una string que será parte de su nombre de archivo

    :param path_files: Path donde encontraremos el/los archivos
    :param part_name_to_find: Parte del nombre del archivo. Para que encontremos todos los archivos que contienen
                                esta string.
    :param path_to_move: Path donde moveremos el/los archivos encontrados
    :return:
    '''
    print("[INFO] Selected path: " + path_files)
    files_on_path = os.listdir(path_files)
    print(files_on_path)
    print("[INFO] Number of files found: " + str(len(files_on_path)))

    print("[INFO] Moving all files that contains (" + part_name_to_find + ") in his file name")

    c = 0
    for f in files_on_path:
        f1 = path_files + f
        f2 = path_to_move + f

        if f.find(part_name_to_find) > 0:
            os.replace(f1, f2)
            c += 1
    print("[INFO] Moved " + str(c) + " files to " + path_to_move)


# USO
p1 = "../dataset/rgbd-scenes-v2/imgs/scene_14/"
p2 = "depth"
p3 = "../dataset/depth-all/"

move_files_by_string(p1, p2, p3)