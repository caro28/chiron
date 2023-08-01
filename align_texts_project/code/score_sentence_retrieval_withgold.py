#!/usr/bin/env python

import pandas as pd
import numpy as np

import argparse

from ast import literal_eval
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def read_alignments(fin):
    """
    function built by vecalign. see:
    https://github.com/caro28/vecalign/blob/master/dp_utils.py
    """
    
    alignments = []
    with open(fin, 'rt', encoding="utf-8") as infile:
        for line in infile:
            fields = [x.strip() for x in line.split(':') if len(x.strip())]
            if len(fields) < 2:
                raise Exception('Got line "%s", which does not have at least two ":" separated fields' % line.strip())
            try:
                src = literal_eval(fields[0])
                tgt = literal_eval(fields[1])
            except:
                raise Exception('Failed to parse line "%s"' % line.strip())
            alignments.append((src, tgt))

    # I know bluealign files have a few entries entries missing,
    #   but I don't fix them in order to be consistent previous reported scores
    return alignments

def build_gold_src2tgt_ids_dict(gold_alignments):
    # move gold aligns to dict: greek gold src id to gold target id
    gold_el_src_id_2_en_tgt_id = defaultdict(set)
    for gold_src, gold_tgt in gold_alignments:
        for gold_src_id in gold_src:
            for gold_tgt_id in gold_tgt:
                gold_el_src_id_2_en_tgt_id[gold_src_id].add(gold_tgt_id)
    return gold_el_src_id_2_en_tgt_id

def compute_sim_matrix(src_embeds, tgt_embeds):
    # using cosine sim
    return cosine_similarity(src_embeds, tgt_embeds)
    # using dot product
    # return np.matmul(grk_embeds, np.transpose(tgt_embeds))

def get_data_for_metrics(sim_matrix, ground_truth_aligns):
    # counters for metrics
    top_1 = 0
    top_3 = 0
    top_5 = 0
    top_10 = 0

    # list to save idx and top score of chunks not in top 10
    not_top_10 = []

    # build dict of gold src id to tgt id dict
    gold_src_2_tgt_id_dict = build_gold_src2tgt_ids_dict(ground_truth_aligns)

    for idx, vector in enumerate(sim_matrix):
        print(f"processing vector {idx}")
        # get true score - find idx using grd truth alignments
        true_score_idx_lst = gold_src_2_tgt_id_dict[idx]
        true_scores_lst = []
        for true_score_idx in true_score_idx_lst:
            true_scores_lst.append(vector[true_score_idx])
        print(f"true scores: {true_scores_lst}")
        print(f"there are {len(true_scores_lst)} true scores for vector {idx}")

        # sort vector
        sorted_sim_vec = np.flip(np.sort(vector))
        # print(f"top score: {sorted_sim_vec[0]}")
        print(f"top 10 scores: {sorted_sim_vec[0:10]}")

        # check if true_scores are in top 1
        for true_score in true_scores_lst:
            if true_score >= sorted_sim_vec[0]:
                top_1 += 1/len(true_scores_lst)
                top_3 += 1/len(true_scores_lst)
                top_5 += 1/len(true_scores_lst)
                top_10 += 1/len(true_scores_lst)
                print("in top 1")
                continue

            elif true_score >= sorted_sim_vec[2]:
                top_3 += 1/len(true_scores_lst)
                top_5 += 1/len(true_scores_lst)
                top_10 += 1/len(true_scores_lst)
                print("in top 3")
                continue

            elif true_score >= sorted_sim_vec[4]:
                top_5 += 1/len(true_scores_lst)
                top_10 += 1/len(true_scores_lst)
                print("in top 5")
                continue

            elif true_score >= sorted_sim_vec[9]:
                top_10 += 1/len(true_scores_lst)
                print("in top 10")
                continue

            else:
                # save chunk (idx, true score, top score)
                data_dict = {}
                data_dict['idx'] = idx
                data_dict['true_scores'] = true_scores_lst
                data_dict['top_score'] = sorted_sim_vec[0]

                idx_topscore = np.where(vector==sorted_sim_vec[0])[0][0]
                data_dict['idx_topscore'] = idx_topscore

                score_differences = []
                for true_score in true_scores_lst:
                    score_differences.append(sorted_sim_vec[0] - true_score)
                data_dict['difference'] = score_differences
                not_top_10.append(data_dict)
                print("not in top 10")
    
    print("done")
    
    print(f"num top 1: {top_1}")
    print(f"num top 10: {top_10}")
  
    return top_1, top_3, top_5, top_10, not_top_10

def get_sim_stats(sim_matrix, gold_alignments):
    # get number of chunks with correct alignment in top scores
    top_1, top_3, top_5, top_10, not_top_10 = get_data_for_metrics(sim_matrix, gold_alignments)
    # calculate percentages
    perc_top1 = top_1/sim_matrix.shape[0]
    perc_top10 = top_10/sim_matrix.shape[0]

    return perc_top1, perc_top10, not_top_10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sentence-level retrieval task for baseline evaluation of LaBSE vs LASER")
    
    parser.add_argument("--src_embed", type=str, required=True,
    help="Path to source embeddings in binary file")

    parser.add_argument("--tgt_embed", type=str, required=True,
    help="Path to tgt embeddings in binary file")

    parser.add_argument("--dim_embed", type=int, required=True,
    help="Dimension of embeddings (768 for LaBSE and 1024 for LASER)")

    parser.add_argument("--gold", type=str, required=True,
    help="Path to gold alignments in plain file. Alignments are at sentence-level, in [1] : [1 or many] format")

    args = parser.parse_args()

    # load and resize embeddings
    dim = args.dim_embed
    src_embeddings = np.fromfile(args.src_embed, dtype=np.float32, count=-1)
    tgt_embeddings = np.fromfile(args.tgt_embed, dtype=np.float32, count=-1)
    src_embeddings.resize(src_embeddings.shape[0] // dim, dim)
    tgt_embeddings.resize(tgt_embeddings.shape[0] // dim, dim)

    print(src_embeddings.shape)
    print(tgt_embeddings.shape)

    # compute similarity matrix
    sim_matrix = compute_sim_matrix(src_embeddings, tgt_embeddings)

    # load ground truth alignments
    gold_path = args.gold
    gold_alignments_src2tgt = read_alignments(gold_path)

    # from vecalign: convert to sets, remove alignments empty on both sides
    gold_alignments_src2tgt = set([(tuple(x), tuple(y)) for x, y in gold_alignments_src2tgt if len(x) or len(y)])

    # get evaluation stats
    perc_top1, perc_top10, not_top_10 = get_sim_stats(
        sim_matrix, gold_alignments_src2tgt)
    
    print(f"top 1 percentage is {perc_top1}")
    print(f"top 10 percentage is {perc_top10}")



