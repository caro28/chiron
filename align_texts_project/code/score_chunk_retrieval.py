#!/usr/bin/env python

import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


#===================== EVALUATION =========================#
def compute_sim_matrix_dotprod(grk_embeds, tgt_embeds):
    return np.matmul(grk_embeds, np.transpose(tgt_embeds))

def compute_sim_matrix_cosine(src_embeds, tgt_embeds):
    # using cosine sim
    return cosine_similarity(src_embeds, tgt_embeds)

def get_data_for_metrics(sim_matrix):
  # counters for metrics
  top_1 = 0
  top_3 = 0
  top_5 = 0
  top_10 = 0

  # list to save idx and top score of chunks not in top 10
  not_top_10 = []

  for idx, vector in enumerate(sim_matrix):
    print(f"processing vector {idx}")
    # get true score, or element on diagonal
    true_score = vector[idx]
    # print(f"true score: {true_score}")

    # sort vector
    sorted_sim_vec = np.flip(np.sort(vector))
    # print(f"top score: {sorted_sim_vec[0]}")

    # check if true_score is in top 1
    if true_score >= sorted_sim_vec[0]:
      top_1 += 1
      top_3 += 1
      top_5 += 1
      top_10 += 1
    #   print("in top 1")
      continue

    elif true_score >= sorted_sim_vec[2]:
      top_3 += 1
      top_5 += 1
      top_10 += 1
    #   print("in top 3")
      continue
    
    elif true_score >= sorted_sim_vec[4]:
      top_5 += 1
      top_10 += 1
    #   print("in top 5")
      continue
    
    elif true_score >= sorted_sim_vec[9]:
      top_10 += 1
    #   print("in top 10")
      continue
    
    else:
      # save chunk (idx, true score, top score)
      data_dict = {}
      data_dict['idx'] = idx
      data_dict['true_score'] = true_score
      data_dict['top_score'] = sorted_sim_vec[0]

      idx_topscore = np.where(vector==sorted_sim_vec[0])[0][0]
      data_dict['idx_topscore'] = idx_topscore
      
      data_dict['difference'] = sorted_sim_vec[0] - true_score
      not_top_10.append(data_dict)
    #   print("not in top 10")
    
  print("done")
  
  return top_1, top_3, top_5, top_10, not_top_10

def get_sim_stats(sim_matrix):
  # get number of chunks with correct alignment in top scores
  top_1, top_3, top_5, top_10, not_top_10 = get_data_for_metrics(
      sim_matrix)

  # calculate percentages
  perc_top1 = top_1/len(sim_matrix)
  perc_top10 = top_10/len(sim_matrix)

  return perc_top1, perc_top10, not_top_10


def get_top_5_errors_idx(not_top_10):
    # get list of differences between true and top scores
    differences_lst = []
    for chunk_dict in not_top_10:
        differences_lst.append(
            (
                float(chunk_dict['idx']),
                float(chunk_dict['difference']),
                float(chunk_dict['idx_topscore']))
        )
    
    differences = []
    for triple in differences_lst:
        differences.append(triple[1])
    
    # get top 5 differences
    differences_sorted = sorted(differences)
    
    first_max_diff_idx = differences.index(differences_sorted[-1])
    second_max_diff_idx = differences.index(differences_sorted[-2])
    third_max_diff_idx = differences.index(differences_sorted[-3])
    fourth_max_diff_idx = differences.index(differences_sorted[-4])
    fifth_max_diff_idx = differences.index(differences_sorted[-5])
    
    idx_list = []
    idx_list.append(first_max_diff_idx)
    idx_list.append(second_max_diff_idx)
    idx_list.append(third_max_diff_idx)
    idx_list.append(fourth_max_diff_idx)
    idx_list.append(fifth_max_diff_idx)
    return idx_list, differences_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score chunk-level retrieval experiments using chunk embeddings")
   
    parser.add_argument("--src_embed", type=str, required=True, help="Path to source embeddings in binary file")
                       
    parser.add_argument("--tgt_embed", type=str, required=True, help="Path to tgt embeddings in binary file")

    parser.add_argument("--dim_embed", type=int, required=True, help="Dimension of embeddings (768 for LaBSE and 1024 for LASER)")

    args = parser.parse_args()

    # load and resize embeddings
    dim = args.dim_embed
    src_embeddings = np.fromfile(args.src_embed, dtype=np.float32, count=-1)
    tgt_embeddings = np.fromfile(args.tgt_embed, dtype=np.float32, count=-1)
    src_embeddings.resize(src_embeddings.shape[0] // dim, dim)
    tgt_embeddings.resize(tgt_embeddings.shape[0] // dim, dim)

    # get sim matrix
    sim_matrix = compute_sim_matrix_cosine(src_embeddings, tgt_embeddings)

    perc_top1, perc_top10, not_top_10 = get_sim_stats(sim_matrix)

    # write results to terminal
    print(f"percent of chunks with true in top 1 sim score: {perc_top1}\n")
    print(f"percent of chunks with true in top 10 sim score: {perc_top10}\n")

    # save dict with data on chunks not in top 10
    json_dict = {}
    json_dict['data'] = not_top_10
    # with open('/home/craig.car/spring2023/results/thuc_eng6_not_top_10.json', 'w') as fp:
    #     json.dump(json_dict, fp)