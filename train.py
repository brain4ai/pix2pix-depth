import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
import config


BATCH_SIZE = config.BATCH_SIZE
IMG_WIDTH = config.IMG_WIDTH
IMG_HEIGHT = config.IMG_HEIGHT
PATH = config.BASE_DATASET_PATH

# Decidimos si entrenamos desde cero o cargamos el ultimo checkpoint
load_ckpt = config.LOAD_CHECKPOINTS


def load(image_file):
    '''
    Conociendo el dataset, vemos que las imagenes estan formadas por dos imagenes.
    Una a continuación de la otra por tanto, leemos la imagen
    y la dividimos en 2 de 256 de ancho
    '''

    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)

    # Obtenemos el ancho y dividimos entre 2
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# Cargamos una pareja de imagenes para ver que funciona correctamente
inp, re = load(PATH + 'train/00000-color-s1.png')
# Normalizamos las imagenes para que las muestre correctamente matplotlib
plt.figure()
plt.imshow(inp/255.0)
plt.show()

plt.figure()
plt.imshow(re/255.0)
plt.show()


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


'''
Definimos los metodos de random jitter como se indican en el paper para el  data augmentation
-resize
-random crop
-random flip
'''
@tf.function()
def random_jitter(input_image, real_image):
    # resize de 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # recorte aleatorio dejandolo en 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # aplicamos giros random
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

# Creamos 4 imagenes para ver si funciona correctamente
plt.figure(figsize=(6, 6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    plt.subplot(2, 2, i + 1)
    plt.imshow(rj_inp / 255.0)
    plt.axis('off')
plt.show()


# Debemos observar que finalmente pasaremos las imagenes normalizadas a los modelos
# haciendo uso de la funcion normalize() anteriormente definida
def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# Creamos nuestros datos de train, buscamos cualquier archivo en el path con
# extension .png
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.png')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Creamos nuestros datos de test
test_dataset = tf.data.Dataset.list_files(PATH+'test/*.png')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


'''
Creamos el GENERADOR

-La aquitectura del Generador es una U-Net modificada.
-Cada bloque del encoder es (Conv -> Batchnorm -> Leaky ReLU)
-Cada bloque del decoder es (T.Conv -> Batchnorm -> Dropout(aplicado a los 3 primeros bloques) -> ReLU)
-Tendremos las skip connections entre encoder y decoder (como en las U-Net).
'''
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
down_result = down_model(tf.expand_dims(inp, 0))
print("[INFO] downsample_shape:")
print (down_result.shape)

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
up_result = up_model(down_result)
print("[INFO] upsample_shape:")
print (up_result.shape)

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

    # Creamos las skip connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
generator.summary()


'''
Generator loss

Explicación : https://brain4ai.wordpress.com/2020/09/22/pix2pix-depth-1/
'''
LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    # loss_object se define mas adelante en el codigo
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


'''
Creamos el Discriminador

-El Discriminador es una PatchGAN.
-Cada bloque del Discriminador es (Conv -> BatchNorm -> Leaky ReLU)
-Las dimensiones de la salida tras la ultima capa son (batch_size, 30, 30,1)
-Cada patch de 30x30 de la salida clasificará una porcion de imagen de 70x70 de la imagen de entrada
(como ocurre en la arquitectura PatchGAN)
-El Discriminador recibe 2 entradas.
    -Imagen de entrada y target imagen, la cual deberia ser clasificada como real 
    -Imagen de entrada e imagen generada por el generador, la cual deberia ser clasificada como fake
    -Estamos concatenando estas dos inputs en el codigo x = tf.keras.layers.concatenate([inp, tar])
'''

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
discriminator.summary()

'''
Discriminator loss

Explicación : https://brain4ai.wordpress.com/2020/09/22/pix2pix-depth-1/
'''
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

'''Definimos los Optimizadores'''
generator_optimizer = tf.keras.optimizers.Adam(config.LR, beta_1=config.MOMENTUM)
discriminator_optimizer = tf.keras.optimizers.Adam(config.LR, beta_1=config.MOMENTUM)

'''Checkpoint-saver'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

'''Load checkpoint comprobation'''
if load_ckpt:
    print("[INFO] Cargando el modelo...")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("[INFO] Modelo cargado correctamente")


'''
-Generamos y salvamos algunas imagenes durante el entrenamiento

-Pasaremos algunas imagenes del test dataset al Generador y este
tratara de traducirlas

-Las imagenes se mostrarán en pantalla durante 3 segundos, se cerrarán y continuara el entrenamiento 
'''

n_IM = 0
def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # escalamos los valores de los pixeles entre [0, 1] para dibujarlos
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    global n_IM

    # Añadimos esto para no tener que cerrar manualmente los plot
    # los 3seg. se estarían sumando al tiempo que tarda en ejecutarse cada epoca
    plt.savefig('imagen_epoca_{:04d}.png'.format(n_IM))
    n_IM += 1
    plt.show(block=False)
    plt.pause(3)  # Esperamos 3s
    plt.close()

# Comprobamos que todo va bien
for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)

'''
Training

-Para cada imagen input generamos una imagen output
-Al discriminador le pasaremos la input_image y la generated image como primera pareja de entrada.
Como segunda pareja de entrada le pasaremos input_image y la target_image
-Despues, calculamos las perdidas en Generador y Discriminador
-Continuamos calculando los gradientes y aplicando estos a los optimizadores
-Por ultimo escribimos los log para que podamos seguir la evolucion del entrenamiento
'''

EPOCHS = config.NUM_EPOCHS

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

'''
Training loop:

-Iteramos durante el numero de epocas especificado
-Pintaremos un '.' cada 100 imagenes
-Salvaremos un checkpoint cada 50 epocas, especificadas en config.py
'''
def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # checkpoint 50 epochs
        if (epoch + 1) % config.SAVE_AFTER_N_EPOCHS == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)

''' Lanzamos el training loop '''
print("[INFO] Entrenando...")
fit(train_dataset, EPOCHS, test_dataset)
print("[INFO] Entrenamineto terminado correctamente")