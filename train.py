# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 09:14:25 2020

@author: Himansh
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
from IPython import display
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, LeakyReLU, Dropout, ReLU, Input, Concatenate, Activation, MaxPooling2D
from tensorflow import random_normal_initializer
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.constraints import Constraint
import argparse
from tensorflow.keras.regularizers import l2
##Parsing command line arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-epochs", "--num_epochs", required = True, help = "Input the number of epochs to be trained")
# ap.add_argument("-dataset", "--dataset_directory", required = True, help = "Input the dataset directory")
# ap.add_argument("-gen", "generator_model", required = True, help = "Choose between u-net generator or Fully Convolutional Densenet")
# ap.add_argument("-bs", "--batch_size", required = True, help = "Enter the batch size, default is 1")
# ap.add_argument("-sd", "--model_save", required = True, help = "Enter the directory to save the model")
# ap.add_argument("-freq", "--save_frequency", required = True, help = "Enter the model saving frequency, this will save generator and discrminator models after every n epochs")
# ap.add_argument("-df", "--disc_frequency", required = True, help = "Number of times discriminator should be updated for every global step")
# ap.add_argument("-cont_loss", "--content_loss", required = True, help = "use content loss for training")
# args = vars(ap.parse_args())
# print(args)

PATH = 'o-haze/'

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

inp, re = load(PATH+'train/1.png')
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp/255.0)
plt.figure()
plt.imshow(re/255.0)

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

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i+1)
  plt.imshow(rj_inp/255.0)
  plt.axis('off')
plt.show()

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

## Crearing the training dataset using tf.data
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.png')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

## Creating testing dataset
test_dataset = tf.data.Dataset.list_files(PATH+'test/*.png')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3


from densenet56 import DenseNet
##Creating instance of densenet
generator = DenseNet()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

# gen_output = generator(inp[tf.newaxis,...], training=False)
# plt.imshow(gen_output[0,...])
def wasserstein_loss(y_true, y_predicted):
    """This function defines wasserstein loss for use in training loop"""
    return K.mean(y_true*y_predicted)

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    """Defines generator loss function containing wasserstein loss and L1 loss"""
    gan_loss = wasserstein_loss(-tf.ones_like(disc_generated_output), disc_generated_output)

  # content loss
  #cont_loss = content_loss(target, gen_output)

  # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))  

  # Total variation loss
  #tv_loss = tf.image.total_variation(gen_output)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

from tensorflow.keras import backend
class ClipConstraint(Constraint):
    """Defines clip constraint for gradient clipping"""
	# set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
 
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


def downsample(filters, size, apply_batchnorm=True):

  init = random_normal_initializer(0., 0.02)
  const = ClipConstraint(0.01)

  result = Sequential()
  result.add(Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=init, kernel_constraint=const,use_bias=False))

  if apply_batchnorm:
    result.add(BatchNormalization())

  result.add(LeakyReLU())

  return result

  
def Discriminator():
  init= tf.random_normal_initializer(0., 0.02)  
  const = ClipConstraint(0.01)
  inp = Input([256, 256, 3])
  tar = Input([256, 256, 3])

  merged = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(merged) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = Conv2D(512, 4, strides=1,
                                kernel_initializer=init,kernel_constraint=const ,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = BatchNormalization()(conv)

  leaky_relu = LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = Conv2D(1, 4, strides=1,
                                kernel_initializer=init,kernel_constraint=const)(zero_pad2) # (bs, 30, 30, 1)

  discriminator_out = Activation('linear')(last)
  model =Model([inp,tar], discriminator_out)
  return model


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
# plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()

def discriminator_loss(disc_real_output, disc_generated_output):
    """Defines discriminator loss"""
    real_loss = wasserstein_loss(-tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = wasserstein_loss(tf.ones_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

##Defining the optimizers
generator_optimizer = tf.keras.optimizers.RMSprop(lr = 0.00005)
discriminator_optimizer = tf.keras.optimizers.RMSprop(lr = 0.00005)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15,15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  
for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)
    
EPOCHS = 150

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

UPDATE_DISC = 1

def train_step(input_image, target, epoch):
  for _ in range(UPDATE_DISC):
    with tf.GradientTape() as disc_tape:
      gen_output = generator(input_image, training=True)
      disc_real_output = discriminator([input_image, target], training=True)
      disc_generated_output = discriminator([input_image, gen_output], training=True)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
  with tf.GradientTape() as gen_tape:
    gen_output = generator(input_image, training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
  generator_gradients = gen_tape.gradient(gen_total_loss,
                                        generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

#os.mkdir('saved_models')

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      
      generator.save('o-haze_models/model_epoch{}.h5'.format(epoch))
      discriminator.save('o-haze_models/disc_epoch{}.h5'.format(epoch))
    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)

## Training the model
fit(train_dataset, EPOCHS, test_dataset)



