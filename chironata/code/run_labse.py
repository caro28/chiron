#!/usr/bin/env python

import sys
import numpy as np
from timeit import default_timer as timer
from sentence_transformers import SentenceTransformer

        
#===================== FUNCTIONS =======================#
def load_txt_as_lst(path_in):
  output_lst = []
  with open(path_in, "rt") as f:
    for line in f:
      output_lst.append(line)
  return output_lst

def build_embeddings_huggingface(sentences_lst, model_):
  embeddings = model_.encode(sentences_lst)
  return embeddings

def write_to_file(embeds_lst, file_out):
  embeds_lst = np.array(embeds_lst)
  with open(file_out, 'w') as f:
    embeds_lst.tofile(f)
# =========================================================#


if __name__ == "__main__":
    # load file to embed
    file = sys.argv[1]
    txt_lst = load_txt_as_lst(file)

    # get embeddings
    model = SentenceTransformer('sentence-transformers/LaBSE')
    labse_embeds = build_embeddings_huggingface(txt_lst, model)
        
    for embed in labse_embeds:
       print("have embed")
