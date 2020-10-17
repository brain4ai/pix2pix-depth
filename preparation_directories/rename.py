import os


def add_string_at_the_end(path_files, new_name):
    '''
    Añadira la string que le pasemos al nombre de archivo renombrandolo.

    :param path_files: path donde se encuentrar los archivos a renombrar
    :param new_name: string que añadiremos al final del nombre de archivo
    :return:
    '''
    print("[INFO] Selected path: " + path_files)
    files_on_path = os.listdir(path_files)
    print("[INFO] Number of files found: " + str(len(files_on_path)))
    print("[INFO] Renaming all files...")

    c = 0
    for f in files_on_path:
        # Detect name and extension
        filename, file_extension = os.path.splitext(f)

        # Create new path and new name
        f1 = path_files + f
        f2 = path_files + filename + new_name + file_extension
        os.rename(f1, f2)
        c += 1

    print("[INFO] Renamed " + str(c) + " files")

# USO
p1 = "../dataset/rgbd-scenes-v2/imgs/scene_01/"
p2 = "-s1"
add_string_at_the_end(p1, p2)