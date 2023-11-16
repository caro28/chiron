#!/usr/bin/env python3

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import argparse
import sys
from collections import defaultdict

import numpy as np

from vecalign_dp_utils import read_alignments

"""
Faster implementation of lax and strict precision and recall, based on
   https://www.aclweb.org/anthology/W11-4624/.

"""


def _precision(goldalign, testalign):
    """
    ********** Original vecalign scoring function. **********
    Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments
    """
    tpstrict = 0  # true positive strict counter
    tplax = 0     # true positive lax counter
    fpstrict = 0  # false positive strict counter
    fplax = 0     # false positive lax counter

    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    # mappings from source test sentence idxs to
    #    target gold sentence idxs for which the source test sentence 
    #    was found in corresponding source gold alignment
    src_id_to_gold_tgt_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_src_id in gold_src:
            for gold_tgt_id in gold_tgt:
                src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)

    for (test_src, test_target) in testalign:
        if (test_src, test_target) == ((), ()):
            continue
        if (test_src, test_target) in goldalign:
            # strict match
            tpstrict += 1
            tplax += 1
        else:
            # For anything with partial gold/test overlap on the source,
            #   see if there is also partial overlap on the gold/test target
            # If so, its a lax match
            target_ids = set()
            for src_test_id in test_src:
                for tgt_id in src_id_to_gold_tgt_ids[src_test_id]:
                    target_ids.add(tgt_id)
            if set(test_target).intersection(target_ids):
                fpstrict += 1
                tplax += 1
            else:
                fpstrict += 1
                fplax += 1

    return np.array([tpstrict, fpstrict, tplax, fplax], dtype=np.int32)

def _new_strict(goldalign, testalign):
    """
    New scoring function: strict match at chunk level, after merging predicted alignments
    """
    tpstrict = 0  # true positive strict counter
    tplax = 0     # true positive lax counter
    fpstrict = 0  # false positive strict counter
    fplax = 0     # false positive lax counter
    fnlax = 0     ##### new false negative lax counter #####
    fnstrict = 0  ##### new false negative strict counter #####
    
    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    src_id_to_test_tgt_ids = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_src_id in test_src:
            for test_tgt_id in test_tgt:
                src_id_to_test_tgt_ids[test_src_id].add(test_tgt_id)
    
    tgt_id_to_test_src_ids = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_tgt_id in test_tgt:
            for test_src_id in test_src:
                tgt_id_to_test_src_ids[test_tgt_id].add(test_src_id)

    supersets_match = []
    goldaligns_missed = []
    supersets_notmatch = []
    for (gold_src, gold_target) in goldalign:
        if (gold_src, gold_target) == ((), ()):
            continue

        #### Changed strict scoring: increment by num of tp #####
        if (gold_src, gold_target) in testalign:
            tpstrict += 1
            print(f"strict match is {(gold_src, gold_target)}")
        
        else:
            new_test_src_ids = set()
            new_test_tgt_ids = set()
            # for every id in goldalign, get corresponding ids in test
            for src_gold_id in gold_src:
                for tgt_id in src_id_to_test_tgt_ids[src_gold_id]:
                    new_test_tgt_ids.add(tgt_id)
                    # visit the new test tgt id: get its corresponding src in test
                    for src_id in tgt_id_to_test_src_ids[tgt_id]:
                        new_test_src_ids.add(src_id)

            # repeat for every tgt id in gold
            for tgt_gold_id in gold_target:
                for src_id in tgt_id_to_test_src_ids[tgt_gold_id]:
                    new_test_src_ids.add(src_id)
                    for tgt_id in src_id_to_test_tgt_ids[src_id]:
                        new_test_tgt_ids.add(tgt_id)
            
            reconstructed_test = (new_test_src_ids,new_test_tgt_ids)
            # convert goldsrc and goldtgt to sets to match reconstructed_test
            if reconstructed_test == (set(gold_src), set(gold_target)):
                print(f"reconstructed_test is {reconstructed_test}")
                print(f"gold align is {(set(gold_src), set(gold_target))}")
                tpstrict += 1
                supersets_match.append(reconstructed_test)
            else:
                # collect missed gold aligns and supersets that do not match
                goldaligns_missed.append((set(gold_src), set(gold_target)))
                supersets_notmatch.append(reconstructed_test)
    
    # print(f'tpstrict is {tpstrict}, fpstrict is {fpstrict}, tplax is {tplax}, fplax is {fplax}, fnlax is {fnlax}, fnstrict is {fnstrict}')
    # return np.array([tpstrict, fpstrict, tplax, fplax, fnlax, fnstrict], dtype=np.int32)

    # return supersets_match for debugging, when running unittests
    print(f"missed goldaligns are {goldaligns_missed}")
    print(f"supersets without matches are: {supersets_notmatch}")
    return tpstrict

def _new_lax(goldalign, testalign):
    """
    New lax scoring: number of overlapping sentences at chunk level, after merging predicted alignments
    """
    tpstrict = 0  # true positive strict counter
    tplax = 0     # true positive lax counter
    fpstrict = 0  # false positive strict counter
    fplax = 0     # false positive lax counter
    fnlax = 0     ##### new false negative lax counter #####
    fnstrict = 0  ##### new false negative strict counter #####
    
    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    # mappings from source test sentence idxs to
    #    target gold sentence idxs for which the source test sentence 
    #    was found in corresponding source gold alignment
    src_id_to_gold_tgt_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_src_id in gold_src:
            for gold_tgt_id in gold_tgt:
                src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)
    
    ##### new dict: maps tgt id to true source id
    tgt_id_to_gold_src_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_tgt_id in gold_tgt:
            for gold_src_id in gold_src:
                tgt_id_to_gold_src_ids[gold_tgt_id].add(gold_src_id)

    for (test_src, test_target) in testalign:
        if (test_src, test_target) == ((), ()):
            continue

        #### Changed strict scoring: increment by num of tp #####
        if (test_src, test_target) in goldalign:
            ##### There's a strict match between true and pred: 
            #####   true src and true tgt ids are tp for strict and lax
            #####       use test_src and test_tgt since they are a match
            tpstrict += len(test_src) + len(test_target)
            tplax += len(test_src) + len(test_target)
        
        else:
            ##### Get gold_all_tgt_ids = {all true_tgt sents for any pred_src sent}
            gold_all_tgt_ids = set()
            for src_test_id in test_src:
                for tgt_id in src_id_to_gold_tgt_ids[src_test_id]:
                    gold_all_tgt_ids.add(tgt_id)

            ##### Get gold_all_src_ids = {all true_src for any pred_tgt sent}
            gold_all_src_ids = set()
            for tgt_test_id in test_target:
                for src_id in tgt_id_to_gold_src_ids[tgt_test_id]:
                    gold_all_src_ids.add(src_id)

            ##### If there is an intersection between true tgt ids and pred tgt ids:
            if set(test_target).intersection(gold_all_tgt_ids):
                ##### 1. Get tplax
                ##### number of pred_src sents in gold_all_src_ids are tp
                tplax += len(set(test_src).intersection(gold_all_src_ids))
                ##### and number of pred_tgt ids in gold_all_tgt_ids are tp
                tplax += len(set(test_target).intersection(gold_all_tgt_ids))
                
                ##### 2. Get fplax and fnlax from elements not in intersection
                #####   Count elements not in tgt intersection
                remainder_tgt = set(test_target) ^ gold_all_tgt_ids
                for element_tgt in remainder_tgt:
                    if element_tgt in gold_all_tgt_ids:
                        ##### tgt is in true but not pred, therefore false negative
                        #####   increment by 1 because will do per element
                        fnlax += 1
                    else:
                        ##### tgt is not in true so is in pred, and therefore false positive
                        fplax += 1
                #####   Count elements not in src intersection
                remainder_src = set(test_src) ^ gold_all_src_ids
                for element_src in remainder_src:
                    if element_src in gold_all_src_ids:
                        ##### src is in true but not pred, therefore false negative
                        fnlax += 1
                    else:
                        ##### src is not in true so is in pred, and therefore false positive
                        fplax += 1

                ##### 3. Get fpstrict
                #####   No strict match with true and there's an intersection: all pred are fp
                fpstrict += len(test_src) + len(test_target)

                ##### 4. Get fnstrict
                #####   No strict match with true and there's an intersection: all true are fn
                fnstrict += len(gold_all_src_ids) + len(gold_all_tgt_ids)

            ##### If there's no intersection between true and pred:
            else:
                ##### all pred are fpstrict and fplax
                fpstrict += len(test_src) + len(test_target)
                fplax += len(test_src) + len(test_target)

                ##### all true are fnstrict and fnlax
                fnstrict += len(gold_all_src_ids) + len(gold_all_tgt_ids)
                fnlax += len(gold_all_src_ids) + len(gold_all_tgt_ids)

    print(f'tpstrict is {tpstrict}, fpstrict is {fpstrict}, tplax is {tplax}, fplax is {fplax}, fnlax is {fnlax}, fnstrict is {fnstrict}')
    return np.array([tpstrict, fpstrict, tplax, fplax, fnlax, fnstrict], dtype=np.int32)

def score_multiple(gold_list, test_list, value_for_div_by_0=0.0):
    """
    Changes from original Vecalign function:
    1. Lax scores come from _new_lax() only
    2. Adds new accuracy score from _new_strict()
    """
    # accumulate counts for all gold/test files
    pcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    for goldalign, testalign in zip(gold_list, test_list):
        pcounts += _precision(goldalign=goldalign, testalign=testalign)
        # recall is precision with no insertion/deletion and swap args
        test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
        gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
        rcounts += _precision(goldalign=test_no_del, testalign=gold_no_del)

    # Compute results
    # pcounts: tpstrict,fnstrict,tplax,fnlax
    # rcounts: tpstrict,fpstrict,tplax,fplax

    if pcounts[0] + pcounts[1] == 0:
        pstrict = value_for_div_by_0
    else:
        pstrict = pcounts[0] / float(pcounts[0] + pcounts[1])

    # if pcounts[2] + pcounts[3] == 0:
    #     plax = value_for_div_by_0
    # else:
    #     plax = pcounts[2] / float(pcounts[2] + pcounts[3])

    if rcounts[0] + rcounts[1] == 0:
        rstrict = value_for_div_by_0
    else:
        rstrict = rcounts[0] / float(rcounts[0] + rcounts[1])

    # if rcounts[2] + rcounts[3] == 0:
    #     rlax = value_for_div_by_0
    # else:
    #     rlax = rcounts[2] / float(rcounts[2] + rcounts[3])

    if (pstrict + rstrict) == 0:
        fstrict = value_for_div_by_0
    else:
        fstrict = 2 * (pstrict * rstrict) / (pstrict + rstrict)

    # if (plax + rlax) == 0:
    #     flax = value_for_div_by_0
    # else:
    #     flax = 2 * (plax * rlax) / (plax + rlax)
    
    ############ new scores added below ####################
    # new strict score
    # new counters for accuracy score
    num_correct = 0
    accuracy_score = 0
    for goldalign, testalign in zip(gold_list, test_list):
        num_correct += _new_strict(goldalign, testalign)
        accuracy_score += num_correct/len(goldalign)

    # new lax score
    pcounts_newlax = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    # rcounts = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    for goldalign, testalign in zip(gold_list, test_list):
        pcounts_newlax += _new_lax(goldalign=goldalign, testalign=testalign)

    if pcounts_newlax[2] + pcounts_newlax[3] == 0:
        plax = value_for_div_by_0
    else:
        plax = pcounts_newlax[2] / float(pcounts_newlax[2] + pcounts_newlax[3])

    ##### recall strict and lax: I use new fnlax and fnstrict from pcounts instead of rcounts #####
    if pcounts_newlax[2] + pcounts_newlax[4] == 0:
        rlax = value_for_div_by_0
    else:
        rlax = pcounts_newlax[2] / float(pcounts_newlax[2] + pcounts_newlax[4])

    ##### F1 code unchanged #####
    if (plax + rlax) == 0:
        flax = value_for_div_by_0
    else:
        flax = 2 * (plax * rlax) / (plax + rlax)

    result = dict(recall_strict=rstrict,
                  recall_lax=rlax,
                  precision_strict=pstrict,
                  precision_lax=plax,
                  f1_strict=fstrict,
                  f1_lax=flax,
                  accuracy=accuracy_score)

    return result


def log_final_scores(res):
    print(' ---------------------------------', file=sys.stderr)
    print('|             |  Strict |    Lax  |', file=sys.stderr)
    print('| Precision   |   {precision_strict:.3f} |   {precision_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| Recall      |   {recall_strict:.3f} |   {recall_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| F1          |   {f1_strict:.3f} |   {f1_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| Accuracy    |   {accuracy:.3f} |   "n/a" |'.format(**res), file=sys.stderr)
    print(' ---------------------------------', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        'Compute strict/lax precision and recall for one or more pairs of gold/test alignments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--test', type=str, nargs='+', required=True,
                        help='one or more test alignment files')

    parser.add_argument('-g', '--gold', type=str, nargs='+', required=True,
                        help='one or more gold alignment files')

    args = parser.parse_args()

    if len(args.test) != len(args.gold):
        raise Exception('number of gold/test files must be the same')

    gold_list = [read_alignments(x) for x in args.gold]
    test_list = [read_alignments(x) for x in args.test]

    res = score_multiple(gold_list=gold_list, test_list=test_list)
    log_final_scores(res)


if __name__ == '__main__':
    main()