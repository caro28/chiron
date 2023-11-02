#!/usr/bin/env python

import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

from functions import load_txt_as_lst

#===================== FUNCTIONS =======================#
# def load_txt_as_lst(path_in):
#     output_lst = []
#     with open(path_in, "rt") as f:
#         for line in f:
#             output_lst.append(line)
#     return output_lst

def build_embeddings_huggingface(sentences_lst, model_):
    embeddings = model_.encode(sentences_lst)
    return embeddings

def write_to_file(embeds_lst, file_out):
    embeds_lst = np.array(embeds_lst)
    with open(file_out, 'w') as f:
        embeds_lst.tofile(f)
# =========================================================#

if __name__ == "__main__":
    print("&&&")
    # load file to embed
    file = sys.argv[1]
    txt_lst = load_txt_as_lst(file)

    # get embeddings
    print("loading model")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    labse_embeds = build_embeddings_huggingface(txt_lst, model)
    print("built embeds")
    prefix = os.path.splitext(file)[0]
    path_out = prefix+".emb"

    write_to_file(labse_embeds,path_out)

    # # for _emb.dummy file
    # for embed in labse_embeds:
    #    print("have embed")
