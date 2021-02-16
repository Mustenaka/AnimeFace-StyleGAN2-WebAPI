import sys
sys.path.append("stylegan2")

import tensorflow as tf
import pickle
import numpy as np

import run_projector
import projector
import dnnlib
import dnnlib.tflib as tflib

# 测试代码

tflib.init_tf()

input_sg_name = "stylegan2_animeface_model/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl"

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