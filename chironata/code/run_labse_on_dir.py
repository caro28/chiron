#!/usr/bin/env python

import os, glob
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

from functions import load_txt_as_lst

'''
Build labse embeds for every file in input directory. Uses one slurm call
only to save time by only loading model once.
'''

#===================== FUNCTIONS =======================#
def build_embeddings_huggingface(sentences_lst, model_):
    embeddings = model_.encode(sentences_lst)
    return embeddings

def write_to_file(embeds_lst, file_out):
    embeds_lst = np.array(embeds_lst)
    with open(file_out, 'w') as f:
        embeds_lst.tofile(f)
# =========================================================#

sents_dir = sys.argv[1]
print(f"working on dir {sents_dir}")
dir_out = sys.argv[2]
print(f"dir out is {dir_out}")

# get embeddings
print("loading model")
model = SentenceTransformer('sentence-transformers/LaBSE')

for sents_path in glob.iglob(sents_dir+"*.sents"):
    filename = os.path.splitext(os.path.basename(sents_path))[0]
    if os.path.isfile(dir_out+filename+".emb") == False:
        print(f"working on {filename}")
        txt_lst = load_txt_as_lst(sents_path)
        labse_embeds = build_embeddings_huggingface(txt_lst, model)
        print("built embeds")
        path_out = dir_out+filename+".emb"
        write_to_file(labse_embeds,path_out)
        print(f"wrote to file {path_out}")

    
