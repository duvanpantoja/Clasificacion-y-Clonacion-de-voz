import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Esta GAN esta adaptada para crear modelos que clonen los espectros de voz de un hablante
#de entrada a un hablante objetivo


#Ruta conjunto de datos
PATH = "/content/drive/My Drive/Clonado/Frases Nariñenses OSC_DUV/TRAIN"
#Ruta datos de input
INPATH = PATH + '/Oscar'
#Ruta datos target
OUTPATH = PATH + '/Duvan'
#Ruta checkpoints
#CKPATH = PATH + '/checkpoints'

imgurls = !ls -1 '{INPATH}'  
n = 90
train_n = round(n*1)

#Listado randomizado de imagenes
randurls = np.copy(imgurls)

#np.random.seed(30)
#np.random.shuffle(randurls)

#Particion train y test
tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]


def comillas(array):
  cont = 0
  for i in array:
      array[cont] = i.replace("'","")
      cont += 1
  return array

tr_urls = comillas(tr_urls)
ts_urls = comillas(ts_urls)
print(len(imgurls), len(tr_urls), len(ts_urls))

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

IMG_WIDTH = 256
IMG_HEIGHT = 256

#Escala imagenes 
def resize(inimg, tgimg, height, width):
  inimg = tf.image.resize(inimg, [height, width])
  tgimg = tf.image.resize(tgimg, [height, width])
  return inimg, tgimg

#Normalizar al rango [-1, +1] las imagenes
def normalize(inimg, tgimg):
  inimg = (inimg / 127.5) - 1
  tgimg = (tgimg / 127.5) - 1
  return inimg, tgimg


#Aumento de datos: Cortes random, Flip
def random_jitter(inimg, tgimg):
  inimg, tgimg = resize(inimg, tgimg, 256, 256)
  stacked_image = tf.stack([inimg, tgimg], axis=0)
  cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH])
  inimg, tgimg = cropped_image[0], cropped_image[1]

  if tf.random.uniform(()) > 0.5:
    inimg = tf.image.flip_left_right(inimg)
    igimg = tf.image.flip_left_right(tgimg)

  return inimg, tgimg

def load_image(filename, augment=True):
  inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + '/' + filename)), tf.float32)
  tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH + '/' + filename)), tf.float32)
  inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)
  inimg, tgimg = normalize(inimg , tgimg)

  if augment:
    inimg, tgimg = random_jitter(inimg, tgimg)
  
  return inimg, tgimg

#Funciones que se encargan de cargar las imagenes
def load_train_image(filename): 
  return load_image(filename, False)

def load_test_image(filename):
  return load_image(filename, False)

#Llamamos la funcion load y retorna la imagen del data y target 
#con todos los preprocesos definidos anteriormente

#Genera un dataset a partir de un listado de datos que le suministro
#En esta seccion por medio del nombre de la imagen se extrae la imagen y se le aplica el 
#preproceso que se encuentra en la funcion load_train_image

train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset = test_dataset.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(1)
tr_urls.dtype

from tensorflow.keras.layers import *
from tensorflow.keras import *

#El paper de Pix2Pix menciona que se aplica una capa Convolucional-BatchNorm y ReLu
#Este es un bloque ENCODER  que representa estas tres capas
#Se esta generando un downsample, es decir realiza una compresion de la imagen
def downsample(filters, apply_batchnorm=True):

  result = Sequential()
  initializer = tf.random_normal_initializer(0, 0.02)

  #####Capa convolucional#####
  result.add(Conv2D(filters,
                    kernel_size=4,
                    strides=2, 
                    padding="same",
                    kernel_initializer=initializer,
                    use_bias=not apply_batchnorm))
  
  #####Capa de BatchNorm#### 
  if apply_batchnorm:
    result.add(BatchNormalization())

  ####Capa de activación#####
  result.add(LeakyReLU())

  return result
downsample(64)

#Bloque DECODER, lo que realiza es que regresa la imagen conprimida a su tamaño original
def upsample(filters, apply_dropout=False):

  result = Sequential()
  initializer = tf.random_normal_initializer(0, 0.02)

  ####Capa convolucional####
  result.add(Conv2DTranspose(filters,
                            kernel_size=4,
                            strides=2,
                            padding="same",
                            kernel_initializer=initializer,
                            use_bias=False))
  
  ####Capa de BatchNora####
  result.add(BatchNormalization())

  ####Capa de Dropout####
  if apply_dropout:
    result.add(Dropout(0.5))

  ####Capa de activación####
  result.add(ReLU())

  return result
upsample(64)

def Generator():
  
  #En esta capa de entrada se le especifica las dimensiones de la serie de datos que se va a ingresar
  inputs = tf.keras.layers.Input(shape=[None,None,3])

  down_stack = [
    downsample(64, apply_batchnorm=False),  #(bs, 128, 128, 64)
    downsample(128),                        #(bs, 64,  64,  128)
    downsample(256),                        #(bs, 32,  32,  256)
    downsample(512),                        #(bs, 16,  16,  512)
    downsample(512),                        #(bs, 8,   8,   512)
    downsample(512),                        #(bs, 4,   2,   512)
    downsample(512),                        #(bs, 2,   2,   512)        
    downsample(512),                        #(bs, 1,   1,   512)           
  ]
  up_stack = [
    upsample(512, apply_dropout=True),     #(bs, 2,   2,   1024)     
    upsample(512, apply_dropout=True),     #(bs, 4,   4,   1024)
    upsample(512, apply_dropout=True),     #(bs, 8,   8,   1024)
    upsample(512),                         #(bs, 16,  16,  1024)
    upsample(256),                         #(bs, 32,  32,  512)
    upsample(128),                         #(bs, 64,  64,  256)    
    upsample(64),                          #(bs, 128, 128, 128)          
  ]

  # Ultima capa que devuelve la imamgen a sus dimensiones originales, cambia su activacion
  initializer = tf.random_normal_initializer(0, 0.02)
  last = Conv2DTranspose(filters = 3,
                         kernel_size = 4,
                         strides = 2,
                         padding = "same",
                         kernel_initializer = initializer,
                         activation = "tanh")
  
  #En primer lugar conecta todas las capas tanto del Encoder como Decoder que se realiza en los ciclos For
  #y para que sea una U-Net necesita de las skip connections y las realiza con la variable s y sk

  x=inputs
  s=[]
  concat = Concatenate()

  for down in down_stack:
    x = down(x)
    s.append(x)

  s = reversed(s[:-1])
  for up, sk in zip(up_stack, s):
    x = up(x)
    x = concat([x, sk])

  last = last(x)

  return Model(inputs=inputs, outputs=last)

generator = Generator()

#Al discriminador ingresan la imagen generada y la input, para realizar una evaluacion de que tan acertada es la imagen
def Discriminator():
  ini = Input(shape=[None, None, 3], name="input_img")
  gen = Input(shape=[None, None, 3], name="gener_img")

  con = concatenate([ini, gen])

  initializer = tf.random_normal_initializer(0, 0.02)

  down1 = downsample(64, apply_batchnorm=False)(con)
  down2 = downsample(128)(down1)
  down3 = downsample(256)(down2)
  down4 = downsample(512)(down3)

  last = tf.keras.layers.Conv2D(filters=1,
                                kernel_size=4,
                                strides=1,
                                kernel_initializer=initializer,
                                padding="same")(down4)
  return tf.keras.Model(inputs=[ini, gen], outputs=last)
discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#Funcion de coste adversaria une los resulatados y hace que durante el entrenamiento esten compitiendo

def discriminator_loss(disc_real_output, disc_generated_output):
  
  #Diferencia entre los true por ser real y el detectado por el discriminador
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  
  #Diferencia entre los false por ser generado y el detectado por el discriminador
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output) 

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output),disc_generated_output)
  #mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))  
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss

#Optimizadores y checkpoints
import os 
generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#checkpoint_prefix = os.path.join(CKPATH, "ckpt")
#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 #discriminator_optimizer=discriminator_optimizer,
                                 #generator=generator,
                                 #discriminator=discriminator)

#checkpoint_restore(tf.train.lasted_checkpoint(CKPATH)).assert_consumed()

def generate_images(model, test_input, tar, save_filename=False, display_imgs=True): 
  prediction = model(test_input, training = True)
  if save_filename:
    tf.keras.preprocessing.image.save_img(PATH + '/outputD/OUT/' + save_filename + '.jpg', prediction[0,...])
    tf.keras.preprocessing.image.save_img(PATH + '/outputD/IN/' + save_filename + '.jpg', test_input[0,...])
    tf.keras.preprocessing.image.save_img(PATH + '/outputD/OR/' + save_filename + '.jpg', tar[0,...])

  plt.figure(figsize=(10,10))
  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth','Predicted Image']

  if display_imgs:
    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      plt.imshow(display_list[i]*0.5 + 0.5)
      plt.axis('off')
  plt.show()

@tf.function
def train_step(input_image, target):

  with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:

    output_image = generator(input_image, training=True)

    output_gen_discr = discriminator([output_image, input_image], training=True)
    
    output_trg_discr = discriminator([target, input_image], training=True)

    discr_loss = discriminator_loss(output_trg_discr, output_gen_discr)

    gen_loss = generator_loss(output_gen_discr, output_image, target)


    generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
                                         
    discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

from IPython.display import clear_output

def train(dataset, epochs):
  for epoch in range(epochs):

    imgi= 0 
    for input_image, target in dataset:
      print('epoch' + str(epoch)+ '- train: '+ str(imgi)+ '/'+ str(len(tr_urls)))
      imgi += 1
      train_step(input_image, target)
      clear_output(wait = True)

    imgi = 0
    for inp, tar in test_dataset.take(1):
      generate_images(generator, inp , tar, str(imgi) + '_' + str(epoch), display_imgs= True)
      imgi += 1

#checkpoint de el modelo cada 20 epochs
#    if(epoch + 1 ) % 50 == 0:
#     checkpoint.save(file_prefix = checkpoint_prefix)

#checkpoint_restore(tf.train.lasted_checkpoint(CKPATH)).assert_consumed()

train(train_dataset,300)

generator.save('/content/drive/My Drive/Clonado/Frases Nariñenses OSC_DUV/TRAIN/modelo_Osc_Duv')
