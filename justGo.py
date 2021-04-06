import os
import random
from pathlib import Path

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

import Log.logutil2 as log
logger = log.logs()

##
# 0. Load network snapshots
##
logger.info("Use tensorflow:{}".format(tf.__version__))


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



##
# 1. Plain generation
##

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


# generate face path ----------------------------------------

def generate_modification_example_path():
    """
    generate example path to /generateFace/example/
    """
    base = "./generateFace/example/"
    generate_path = base + "example-modification.jpg"
    return generate_path

def generate_random_path():
    """
    generate ramdom path to /generateFace/random/
    """
    base = "./generateFace/random/"
    name = str(random.randint(0, 99999999)).zfill(8)
    generate_path = base + "r" +name + ".jpg"
    return generate_path


def generate_example_path():
    """
    generate example path to /generateFace/example/
    """
    base = "./generateFace/example/"
    generate_path = base + "example.jpg"
    return generate_path


# user : 10000001
def generate_user_path(user_id):
    """
    generate ramdom path to /generateFace/example/
    """
    base = "./generateFace/user"
    name = str(random.randint(0, 99999999)).zfill(8)
    generate_base_path = base + "/" + str(user_id) +"/"
    if not Path(generate_base_path).is_dir():
        os.makedirs(generate_base_path)
    generate_path = generate_base_path + "u" + name + ".jpg"
    return generate_path


# --------------------------------------------------------------


def gengerate_random_face():
    print("-----generate random face-----")
    # Just build a single image, randomly
    img_pos_path = generate_random_path()
    arrays, images = generate_from_latents([np.random.randn(1, Gs.input_shape[1])], 0.41)
    images[0].save(img_pos_path)


def gengerate_example_face():
    print("-----generate example face-----")
    # Fixed input example, with dlatents as intermediate
    img_pos_path = generate_example_path()
    dlatents = Gs.components.mapping.run(latents_a, None)[0]
    resut = PIL.Image.fromarray(generate_images_from_dlatents(dlatents, 0.7))
    resut.save(img_pos_path)




def gengerate_example_moive():
    """
    For my computer:
        CPU: G4560
        MEM: 12Gib DDR4 2400
        GPU: NVIDIA GTX960M
    It use about 3~4h to generate a moive.
    Maybe use ffmpeg will fast more.
    """
    array_list = generate_from_generator_adaptive(circ_generator)
    clip = moviepy.editor.ImageSequenceClip(array_list, fps=60)
    clip.ipython_display()
    clip.write_videofile("out.mp4")

##
# 3. Projection - encoding images into latent space
##

# Projects an image into dlatent space and returns the dlatents
def encode_image(image, steps=1000, verbose=True):
    image_processed = np.array(copy.deepcopy(image).convert('RGB').resize((512, 512), resample = PIL.Image.LANCZOS)) / 255.0
    image_processed = (image_processed.transpose(2, 0, 1) - 0.5) * 2.0
    image_processed = np.array([image_processed])
    proj.num_steps = steps
    proj.start(image_processed)
    while proj.get_cur_step() < steps:
        if verbose:
            print('\rProjection: Step %d / %d ... ' % (proj.get_cur_step(), steps), end='', flush=True)
        proj.step()
    print('\r', end='', flush=True)
    return proj.get_dlatents()


def generate_animeFaceImage_from_realFace(open_img_pos_path, aroundTime):
    # Note that projection has a random component - if you're not happy with the result, probably retry a few times
    # For best results, probably have a single person facing the camera with a neutral white background
    image = PIL.Image.open(open_img_pos_path)
    proj_dlatents = encode_image(image, aroundTime)
    return proj_dlatents


def save_proj_dlatents_image(proj_dlatents, changeRate, img_pos_path):
    assert changeRate >= 0 and changeRate <= 1, logger.error("Input error,need [0,1],but your is{}".format(changeRate))
    resut = PIL.Image.fromarray(generate_images_from_dlatents(proj_dlatents, changeRate))
    resut.save(img_pos_path)


##
# 3. Modification - prepararing data for training dlatent dirs
##

# Generate samples to pass to the tagger model
def generate_one_for_latentdirs(truncation_psi = 0.7, randomize_noise = True):
    latents = rnd.randn(1, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(latents, None)[0]
    dlatents_trunc = truncate(dlatents, truncation_psi)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    result_image = Gs.components.synthesis.run(
        dlatents_trunc.reshape((-1, 16, 512)),
        randomize_noise = randomize_noise,
        minibatch_size = 1,
        output_transform=fmt
    )[0]
    return latents, dlatents, result_image

# This gets slower as we go so lets go in steps of 100 to see if that alleviates the issue
# of course, this doesn't work, on account of my GPU memory being too small. minibatch 10, maybe?
# I want a ti 2080 11gb quite badly
def generate_many_for_latentdirs(truncation_psi = 0.7, randomize_noise = True):
    latents = rnd.randn(100, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(latents, None)
    dlatents_trunc = truncate(dlatents, truncation_psi)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    result_image = Gs.components.synthesis.run(
        dlatents_trunc.reshape((-1, 16, 512)),
        randomize_noise = randomize_noise,
        minibatch_size = 10,
        output_transform=fmt
    )
    return latents, dlatents, result_image

# Learn directions for tags (binarized) using logistic regression classification
# Thought: Why not apply L1 regu here to enforce sparsity?
# -> I feel this has neither improved nor worsened the situation (default settings)
# -> With really extreme sparsity constraint, it does, not work very well. Is my assumption that dlatents should be
#    sparse wrt tags wrong? Probably is.
# -> For now, plain logistic regression gives best results
def find_direction_binary(dlatents, targets):
    #clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, C=0.001).fit(dlatents, targets)
    clf = LogisticRegression().fit(dlatents, targets)
    return np.repeat(clf.coef_.reshape(1,512), 16, axis = 0)

# Alternate version: Learn directions by fitting a linear regression onto the probabilities
# Questions:
# * Problem should be sparse - Lasso / Elasticnet?
#    -> ElasticNet does empathically Not work, regularizes everything to 0 (lol)
#    -> Same for Lasso :(
#    -> Lower alpha term? omit intercept?
#    -> seems like it would be really useful, though (not to mention faster), since we _expect_ sparsity here.
#    -> retry later with more data?
#    -> oh lol I'm fucking dumb why was I doing this with the inputs for every level. they're identical. there's just 512.
# * Not the most well-posed problem just generally -> really do need more samples. sighs.
# * Conclusion currently: Doesn't work all that well even though in theory it totally should
# * Or may it it would if you did not Fuck It Up you turbo moron. Retrying with not broken labels now.
#    -> suddenly it works great
def find_direction_continuous(dlatents, targets):
    clf = Lasso(alpha=0.01, fit_intercept=False).fit(dlatents, targets)
    if np.abs(np.sum(clf.coef_)) == 0.0:
        clf = Lasso(alpha=0.001, fit_intercept=False).fit(dlatents, targets)
    return  np.repeat(clf.coef_.reshape(1,512), 16, axis = 0)

# Alternate idea: Fit gauss mixture, gradient descend in that
# -> Ooooor, possibly: gauss mixture mapping kind of simultaneous likelihood maximization type stuff?
# -> that sounds super fun lets try it
# -> But first, the Super Baby Version: Just move towards weighted mean
# -> Works okish, but only okayish
def find_distrib_continuous(dlatents, targets):
    tag_mean = np.sum(dlatents * targets.reshape(-1, 1), axis = 0) / np.sum(targets)
    return  np.repeat(tag_mean.reshape(1,512), 16, axis = 0)


##
# 4. Modification - basic
##

with open("tagged_dlatents/tag_dirs_cont.pkl", 'rb') as f:
    tag_directions = pickle.load(f)

def save_modification_example(img_pos_path):
    # Do some modification and save on a grid
    dlatents_gen = Gs.components.mapping.run(latents_a, None)[0]

    im = PIL.Image.new('RGB', (128 * 5, 128 * 5))
    for i in range(0, 5):
        for j in range(0, 5):
            factor_hair = (i / 5.0) * 25.0
            factor_mouth = (j / 5.0) * 25.0
        
            dlatents_mod = copy.deepcopy(dlatents_gen)
            #dlatents_mod = dlatent_pca.transform(dlatents_mod.reshape(1, 16*512)).reshape(16, 512)
            dlatents_mod += -tag_directions["blonde_hair"] / np.linalg.norm(tag_directions["blonde_hair"]) * factor_hair \
                            +tag_directions["black_hair"] / np.linalg.norm(tag_directions["black_hair"]) * factor_hair
            dlatents_mod += -tag_directions["closed_mouth"] / np.linalg.norm(tag_directions["closed_mouth"]) * factor_mouth \
                            +tag_directions["open_mouth"] / np.linalg.norm(tag_directions["open_mouth"]) * factor_mouth
            #dlatents_mod = dlatent_pca.inverse_transform(dlatents_mod.reshape(1, 16*512)).reshape(16, 512)
            
            dlatents_mod_image = generate_images_from_dlatents(dlatents_mod, 0.6)
            im.paste(PIL.Image.fromarray(dlatents_mod_image, 'RGB').resize((128, 128), resample = PIL.Image.LANCZOS), (128 * i, 128 * j))
    im.save(img_pos_path)


# Remove some tags that are not all that helpful
tags_use = list(tag_directions.keys())

tags_use.remove("face")
tags_use.remove("portrait")
tags_use.remove("pillarboxed")
tags_use.remove("letterboxed")
tags_use.remove("frame")
tags_use.remove("border")
tags_use.remove("black_border")
tags_use.remove("close-up")
tags_use.remove("artist_name")

with open("tagged_dlatents/tags_use.pkl", 'wb') as f:
    pickle.dump(tags_use, f)


hair_eyes_only = False
    
tag_len = {}
for tag in tag_directions:
    tag_len[tag] = np.linalg.norm(tag_directions[tag].flatten())

mod_latents = copy.deepcopy(latents_a)
dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0]  

def modify_and_sample(display_psi, move_dic, is_example, proj_dlatents, save_path):
    assert display_psi <= 1 and display_psi >= 0, logger.error("Input error,need [0,1],but your is{}".format(display_psi))
    if is_example:
        dlatents_mod = copy.deepcopy(dlatents_gen)
    else:
        dlatents_mod = proj_dlatents
        
    for tag in move_dic:
        dlatents_mod += tag_directions[tag] * move_dic[tag]  / tag_len[tag] * 25.0

    result = PIL.Image.fromarray(generate_images_from_dlatents(dlatents_mod, truncation_psi = display_psi), 'RGB')
    result.save(save_path)
    result.show()


def init_modif(user_id, change_tag):
    with open("tagged_dlatents/tags_use.pkl", "rb") as f:
        modify_tags = pickle.load(f)
    
    tag_widgets = {}
    for tag in modify_tags:
        tag_widgets[tag] = change_tag[tag]

    save_path = generate_user_path("10000001")
    proj_dlatents = generate_animeFaceImage_from_realFace("userFace/10000001/jige.png",10)
    modify_and_sample(0.6, tag_widgets, False, proj_dlatents, save_path)


if __name__ == '__main__':
    #gengerate_random_face()
    #gengerate_example_data()
    #proj_dlatents = generate_animeFaceImage_from_realFace("userFace/10000001/jige.png", 20)
    #save_path = generate_user_path("10000001")
    #save_proj_dlatents_image(proj_dlatents, 0.65, save_path)
    #mod_path = generate_modification_example_path()
    #save_modification_example(mod_path)

