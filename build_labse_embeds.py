#!/usr/bin/env python

import argparse
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
  parser = argparse.ArgumentParser(description="Build LaBSE embeddings from txt file (overlaps in vec pipeline) and write out as binary file")

  parser.add_argument("--i", type=str, required=True,
  help="Path to file to be embedded (overlaps for vecalign pipeline)")
  
  parser.add_argument("--o", type=str, required=True,
  help="Path out for embeddings file")

  args = parser.parse_args()

  # load overlaps files and convert to list
  print("loading file to embed")
  txt_lst = load_txt_as_lst(args.i)
  print(f"there are {len(txt_lst)} sentences in input file")

  # get embeddings
  print("building embeddings")
  start_loadmodel = timer()
  model = SentenceTransformer('sentence-transformers/LaBSE')
  end_loadmodel = timer()
  run_time_load = int(end_loadmodel-start_loadmodel)
  print(f"loaded model in {run_time_load} seconds")

  start_embed = timer()
  labse_embeds = build_embeddings_huggingface(txt_lst, model)
  end_embed = timer()
  runtime_embed = int(end_embed-start_embed)
  print(f"embeddings file will include {len(labse_embeds)} embeddings")
  print(f"built embeddings in {runtime_embed} seconds")
    
  write_to_file(labse_embeds, args.o)

  print("Sent embeds to file")