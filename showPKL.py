import pickle

import numpy as np

with open("tagged_dlatents/tag_dirs_cont.pkl", 'rb') as f:
    tag_directions = pickle.load(f)
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

    #print(tags_use)
    #print(len(tags_use))


hair_eyes_only = False
    
tag_len = {}
for tag in tag_directions:
    tag_len[tag] = np.linalg.norm(tag_directions[tag].flatten())

print(tag_len)