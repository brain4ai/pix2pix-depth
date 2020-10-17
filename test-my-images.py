import tensorflow as tf
import os
from matplotlib import pyplot as plt
import open3d as o3d
import cv2
import numpy as np
import config

IMG_WIDTH = config.IMG_WIDTH
IMG_HEIGHT = config.IMG_HEIGHT
PATH = config.VALIDATION_OUTPUT_PATH
# Modo pruebas, si quieres cargar imagenes tecleando su path
mis_imagenes = False


'''Creamos el GENERADOR'''

OUTPUT_CHANNELS = 3
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

down_model = downsample(3, 4)

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

up_model = upsample(3, 4)

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()


'''Generator loss'''

LAMBDA = 100
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


'''Build the Discriminator'''

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


'''Discriminator loss'''

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


'''Define the Optimizers'''

generator_optimizer = tf.keras.optimizers.Adam(config.LR, beta_1=config.MOMENTUM)
discriminator_optimizer = tf.keras.optimizers.Adam(config.LR, beta_1=config.MOMENTUM)


'''Checkpoint-saver'''

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


'''Load checkpoint'''

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print("[INFO] Restore Checkpoint Model without any error")


'''Serialize generator'''

# generator.save("./serialized-generator/generator.hdf5")
# print("[INFO] Generator Model serialized OK")


'''Carga de imagenes, tal cual la implementamos en train.py'''

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    print(tf.shape(input_image))
    return input_image, real_image


'''Creamos Resize si hiciese falta. En nuestro caso ya tenemos el dataset con el tamaño correcto'''


'''Creamos Normalize, tal cual lo implementamos en train.py'''

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


'''Creamos la funcion de carga de imagenes definitiva, que envuelve a load y a normalize'''

def load_image_val(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


'''Creamos una funcion para leer imagenes y pasarlas al tipo de datos que utiliza open3d'''

def o3d_read_image(path):
    im1 = cv2.imread(path)
    cv2.imwrite(path, im1)
    image_raw = o3d.io.read_image(path)

    return image_raw


'''Creamos una funcion para crear una nube de puntos a partir de una imagen a color y su mapa de profundidad'''

def image_and_depthmap_to_pointcloud(im_color, im_depth, pcd_name="./TestData.pcd"):
    # Creamos imagen RGBD a partir de imagen a color y mapa de profundidad
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_color, im_depth, depth_scale=255.0, convert_rgb_to_intensity=False)

    # Los parametros intrinsecos nos los debemos inventar puesto que los originales de kinect
    # no nos sirven
    # - Nuestro tamaño de imagen es 256x256
    # - Como focal length cogeremos el original de kinect y lo escalaremos a 680/256 y 480/256, esto es inventado
    # quizas otros valores funcionen incluso mejor
    # - Como pixel central cogeremos 256/2 y 256/2
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(int(256), int(256),
                             float(525.0 / 2.5), float(525.0 / 1.875),
                             float(256 / 2), float(256 / 2))

    # Si queremos usar los parametros intrinsecos que open3d nos proporciona por defecto descomentar a continuacion
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)

    # Creamos la nube de puntos
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Pasamos a numpy array
    xyz = np.asarray(pcd.points)
    xyz_c = np.asarray(pcd.colors)

    # Escalamos a valores mas reales
    xyz = xyz * 1000

    # Filtro por distancia en z
    # Solo si la componente z es mayor que dist la conservamos
    xyz_filtrada = []
    xyz_filtrada_colores = []
    dist = 0.4
    i = 0
    for vector in xyz:
        if vector[2] > dist:
            xyz_filtrada.append(vector)
            xyz_filtrada_colores.append(xyz_c[i])
        i += 1

    # Pasamos a numpy array las nubes filtradas
    xyz_filtrada = np.array(xyz_filtrada)
    xyz_filtrada_colores = np.array(xyz_filtrada_colores)

    print("Cantidad de puntos antes de filtrar por distancia z = " + str(len(xyz)))
    print("Cantidad de puntos despues de filtrar por distancia z = " + str(len(xyz_filtrada)))

    # Pasamos a nube de puntos completando los campos puntos y color
    pcd.points = o3d.utility.Vector3dVector(xyz_filtrada)
    pcd.colors = o3d.utility.Vector3dVector(xyz_filtrada_colores)

    # Quitamos outlier para limpiar imagen
    # Debe haber mas de 100 puntos vecinos en un radio de 1 para considerarse inlier
    #pcd, ind = pcd.remove_radius_outlier(nb_points=100, radius=0.1)
    #pcd = pcd.select_by_index(ind)

    # Salvamos en .pcd - opcional
    o3d.io.write_point_cloud(pcd_name, pcd, write_ascii=True)

    return pcd


'''Creamos el dataset de validacion'''

val_dataset = tf.data.Dataset.list_files(PATH+'*.png')
val_dataset = val_dataset.map(load_image_val)
val_dataset = val_dataset.batch(config.BATCH_SIZE)


'''Creamos generar imagenes y point clouds'''

CUENTA_IM = 0
num_imagenes_a_generar = 1 # Numero de imagenes que cogera de validation para generarnos mapas de profundidad

def generate_images(model, test_input, tar):
    global CUENTA_IM

    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    color_raw = []
    depth_raw = []

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Guardo la imagen de color y la inferida de deph para luego obtener la nube de puntos en color
        if i == 0:
            plt.imsave('imagen_infer_c_{:04d}.png'.format(CUENTA_IM), display_list[i].numpy() * 0.5 + 0.5) # escalamos entre [0, 1]

            color_raw = o3d_read_image('imagen_infer_c_{:04d}.png'.format(CUENTA_IM))

        if i == 2:
            plt.imsave('imagen_infer_d_{:04d}.png'.format(CUENTA_IM), display_list[i].numpy() * 0.5 + 0.5)

            depth_raw = o3d_read_image('imagen_infer_d_{:04d}.png'.format(CUENTA_IM))

        # Mostramos mapa de profundidad inferido
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('imagen_inferida_{:04d}.png'.format(CUENTA_IM))
    plt.show()
    CUENTA_IM += 1

    # Creamos nube de puntos
    pcd = image_and_depthmap_to_pointcloud(color_raw, depth_raw)

    # Rotamos la nube de puntos y la visualizamos
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

'''Ejecutamos algunas predicciones'''

for inp, tar in val_dataset.take(num_imagenes_a_generar):
    generate_images(generator, inp, tar)




'''Bonus'''
'''Modo pruebas, si quieres cargar imagenes tecleando su path ejemplo: ./misfotos/foto1.jpg'''

if mis_imagenes:

    CUENTA_MIS_IM = 0

    def generate_mis_images(model, im_input):
        prediction = model(im_input, training=False)
        plt.figure(figsize=(15, 15))

        display_list = [im_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        plt.subplot(1, 2, 1)
        plt.title(title[0])

        plt.imsave('mis_imagenes_infer_c_{:04d}.png'.format(CUENTA_IM), display_list[0].numpy() * 0.5 + 0.5)
        color_raw = o3d_read_image('mis_imagenes_infer_c_{:04d}.png'.format(CUENTA_IM))

        plt.imshow(display_list[0] * 0.5 + 0.5) # escalamos entre [0, 1]
        plt.subplot(1, 2, 2)
        plt.title(title[1])


        plt.imsave('mis_imagenes_infer_d_{:04d}.png'.format(CUENTA_IM), display_list[1].numpy() * 0.5 + 0.5)
        depth_raw = o3d_read_image('mis_imagenes_infer_d_{:04d}.png'.format(CUENTA_IM))

        plt.imshow(display_list[1] * 0.5 + 0.5) # escalamos entre [0, 1]
        plt.axis('off')

        global CUENTA_MIS_IM
        plt.savefig('mis_imagenes_{:04d}.png'.format(CUENTA_MIS_IM))
        plt.show()
        CUENTA_MIS_IM += 1

        # Creamos nube de puntos
        pcd = image_and_depthmap_to_pointcloud(color_raw, depth_raw)

        # Rotamos la nube de puntos y la visualizamos
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pcd])


    print(" ----- Inference Mode, Mis Imagenes, Modo Pruebas ----- ")
    while True:
        print("Para finalizar teclea: fin")
        print("Introduce el path de tu imagen en .jpg: ")
        file = input()
        if file == "fin" or file == "Fin":
            break
        else:
            # leemos imagen
            im = tf.io.read_file(file)
            im = tf.image.decode_jpeg(im)
            im = im[tf.newaxis, ...]  # Añadimos la dimension extra del batch
            im = tf.cast(im, tf.float32)
            # normalize
            im = (im / 127.5) - 1
            # resize
            res = tf.image.resize(im, [IMG_WIDTH, IMG_HEIGHT], antialias=True)

            generate_mis_images(generator, res)