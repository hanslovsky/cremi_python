#!/usr/bin/python

from cremi import Annotations
from cremi.io import CremiFile
import numpy as np
import random


# Create some dummy annotation data
annotations = Annotations()
for id in [ 0, 1, 2, 3 ]:
    location = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    annotations.add_annotation(id, "presynaptic_site", location)
for id in [ 4, 5, 6, 7 ]:
    location = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    annotations.add_annotation(id, "postsynaptic_site", location)
for (pre, post) in [ (0, 4), (1, 5), (2, 6), (3, 7) ]:
    annotations.set_pre_post_partners(pre, post)
annotations.add_comment(6, "unsure")

# Open a file for writing (deletes previous file, if exists)
file = CremiFile("example.hdf", "w")

# Write the raw volume. This is given here just for illustration. For your 
# submission, you don't need to store the raw data. We have it already.
file.write_raw(np.zeros((10,100,100), dtype=np.uint8), (40.0, 4.0, 4.0))

# Write volumes representing the neuron and synaptic cleft segmentation.
file.write_neuron_ids(np.zeros((10,100,100), dtype=np.uint64), (40.0, 4.0, 4.0), comment="just zeros")
file.write_clefts(np.zeros((10,100,100), dtype=np.uint64), (40.0, 4.0, 4.0), comment="just zeros")

# Write synaptic partner annotations.
file.write_annotations(annotations)

file.close()
