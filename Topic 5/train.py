# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:06:21 2019

@author: Wei-Hsiang, Shen
"""

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time

from model import Generator_Model, Discriminator_Model
from data_generator import Get_DS


matplotlib.use('Agg')

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):1
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.RMSprop(1e-4)
discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-4)

generator = Generator_Model()
discriminator = Discriminator_Model()

noise_dim = 100 # input shape of the generator

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False, so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(11, 11))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :], vmin=0, vmax=1)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)

@tf.function # with this function decorator, tensorflow compiles the function into graph
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    checkpoint_dir = './checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                     discriminator_optimizer = discriminator_optimizer,
                                     generator = generator,
                                     discriminator = discriminator)

    # We will reuse this seed overtime (so it's easier) to visualize progress
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    for epoch in range(epochs): # for each epoch
        start = time.time()

        gen_loss_total = 0
        disc_loss_total = 0
        batch_index = 1

        print("Epoch {}/{}".format(epoch+1, epochs))
        for image_batch in dataset: # for each batch, note that we do not code batch per epoch, the dataset would end if all data is used
            gen_loss, disc_loss = train_step(image_batch)
            batch_index += 1
            gen_loss_total += gen_loss
            disc_loss_total += disc_loss
            if (batch_index % 50) == 0:
                print("Batch: {}".format(batch_index))
        print("Batch: {}/{}".format(batch_index, batch_index))

        # Generate output image at the end of each epoch
        generate_and_save_images(generator, epoch + 1, seed)

        # Save checkpoint per 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            print("Save checkpoint to {}".format(checkpoint_prefix))

        print('Time for epoch {} is {:.1f} sec, gen_loss {:.3f}, disc_loss {:.3f}'.format(epoch + 1, time.time()-start, gen_loss_total, disc_loss_total))

        # Write training log at the end of each epoch
        with open('./checkpoints/train.log', 'a') as the_file:
            the_file.write("Epoch,{},Generator_loss,{:.3f},Discriminator_loss.{:.3f}\n".format(epoch+1, gen_loss_total, disc_loss_total))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)
    checkpoint.save(file_prefix = checkpoint_prefix)
    print("Save checkpoint to {}".format(checkpoint_prefix))

if __name__ == '__main__':
    EPOCHS = 1000
    BATCH_SIZE = 128

    dataset_train, dataset_val = Get_DS(BATCH_SIZE)
    train(dataset_train, EPOCHS)