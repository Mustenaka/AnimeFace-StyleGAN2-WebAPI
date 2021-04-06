import os
import random
import pickle
import numpy as np
import numpy.linalg as la

import PIL.Image
import PIL.ImageSequence

import moviepy
import moviepy.editor
import math
import glob
import csv
from functools import partial
import time
import collections

import sys
sys.path.append("stylegan2")

import run_projector
import projector
import dnnlib
import dnnlib.tflib as tflib

import tensorflow as tf

import keras
from keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

import colorsys
import re
import copy

from IPython.display import display, clear_output
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt
import glob
import gc


# From https://mega.nz/#!PeIi2ayb!xoRtjTXyXuvgDxSsSMn-cOh-Zux9493zqdxwVMaAzp4 - gwern animefaces stylegan2
input_sg_name = "./stylegan2_animeface_model/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl"
#input_sg_name = "./stylegan2_animeface_model/vgg16_anime_perceptual.pkl"

tflib.init_tf()

is_file = os.path.exists(input_sg_name)
print(is_file)


# Load pre-trained network.
with open(input_sg_name, 'rb') as f:
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.    
    _G, _D, Gs = pickle.load(f)
        
# Print network details.
Gs.print_layers()
_D.print_layers()

# For projection
proj = projector.Projector()
proj.set_network(Gs)

# Init randomizer, load latents
rnd = np.random # Could seed here if you want to
latents_a = rnd.randn(1, Gs.input_shape[1])
latents_b = rnd.randn(1, Gs.input_shape[1])
latents_c = rnd.randn(1, Gs.input_shape[1])


if os.path.exists("example_data/latents.npy"):
    latents_a, latents_b, latents_c = np.load("example_data/latents.npy")
np.save("example_data/latents.npy", np.array([latents_a, latents_b, latents_c]))


# "Ellipse around a point but probably a circle since it's 512 dimensions" laten
def circ_generator(latents_interpolate):
    radius = 40.0

    latents_axis_x = (latents_a - latents_b).flatten() / la.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / la.norm(latents_a - latents_c)

    latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
    latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents

# Generate images from a list of latents
def generate_from_latents(latent_list, truncation_psi):
    array_list = []
    image_list = []
    for latents in latent_list:
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=truncation_psi, randomize_noise=False, output_transform=fmt)
        array_list.append(images[0])
        image_list.append(PIL.Image.fromarray(images[0], 'RGB'))
        
    return array_list, image_list

def mse(x, y):
    return (np.square(x - y)).mean()

# Generate from a latent generator, keeping MSE between frames constant
def generate_from_generator_adaptive(gen_func):
    max_step = 1.0
    current_pos = 0.0
    
    change_min = 10.0
    change_max = 11.0
    
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    current_latent = gen_func(current_pos)
    current_image = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
    array_list = []
    
    while(current_pos < 1.0):
        array_list.append(current_image)
        
        lower = current_pos
        upper = current_pos + max_step
        current_pos = (upper + lower) / 2.0
        
        current_latent = gen_func(current_pos)
        current_image = images = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
        current_mse = mse(array_list[-1], current_image)
        
        while current_mse < change_min or current_mse > change_max:
            if current_mse < change_min:
                lower = current_pos
                current_pos = (upper + lower) / 2.0
            
            if current_mse > change_max:
                upper = current_pos
                current_pos = (upper + lower) / 2.0
                
            
            current_latent = gen_func(current_pos)
            current_image = images = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
            current_mse = mse(array_list[-1], current_image)
        print(current_pos, current_mse)
        
    return array_list

# We have to do truncation ourselves, since we're not using the combined network
def truncate(dlatents, truncation_psi, maxlayer = 16):
    dlatent_avg = tf.get_default_session().run(Gs.own_vars["dlatent_avg"])
    layer_idx = np.arange(16)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < maxlayer, truncation_psi * ones, ones)
    return tf.get_default_session().run(tflib.lerp(dlatent_avg, dlatents, coefs))

# Generate image with disentangled latents as input
def generate_images_from_dlatents(dlatents, truncation_psi = 1.0, randomize_noise = True):
    if not truncation_psi is None:
        dlatents_trunc = truncate(dlatents, truncation_psi)
    else:
        dlatents_trunc = dlatents
        
    # Run the network
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    result_image = Gs.components.synthesis.run(
        dlatents_trunc.reshape((-1, 16, 512)),
        randomize_noise = randomize_noise,
        minibatch_size = 1,
        output_transform=fmt
    )[0]
    return result_image


# generate face for random

def generate_random_random_path():
    """
    generate ramdom path to /generateFace/random/
    """
    base = "./generateFace/random/"
    name = str(random.randint(0, 99999999)).zfill(8)
    generate_path = base + name + ".jpg"
    return generate_path

def generate_example_path():
    """
    generate ramdom path to /generateFace/example/
    """
    base = "./generateFace/example/"
    generate_path = base + "example.jpg"
    return generate_path


def gengerate_random_face():
    print("-----generate random face-----")
    # Just build a single image, randomly
    img_pos_path = generate_random_random_path()
    arrays, images = generate_from_latents([np.random.randn(1, Gs.input_shape[1])], 0.41)
    images[0].save(img_pos_path)


def gengerate_example_data(img_pos_path):
    print("-----generate example face-----")
    # Fixed input example, with dlatents as intermediate
    img_pos_path = generate_example_path()
    dlatents = Gs.components.mapping.run(latents_a, None)[0]
    resut = PIL.Image.fromarray(generate_images_from_dlatents(dlatents, 0.7))
    resut.save(img_pos_path)



def gengerate_example_moive():
    array_list = generate_from_generator_adaptive(circ_generator)
    clip = moviepy.editor.ImageSequenceClip(array_list, fps=60)
    clip.ipython_display()
    clip.write_videofile("out.mp4")

if __name__ == '__main__':
    random_path = generate_random_random_path()
    gengerate_random_face(random_path)
    example_path = generate_example_path()
    gengerate_example_data(example_path)