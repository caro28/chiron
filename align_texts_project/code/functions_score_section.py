import re
import json
import stanza
import argparse

import numpy as np
import pandas as pd

from itertools import chain
from ast import literal_eval
from collections import defaultdict

################ Score vecalign results: by prediction ####################

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


def score_vec_rslts_chapter_level(vr_rslts_lst, el_sent2section_dict,
                                 fr_sent2section_dict, fr_extra_section_names):

    tp_strict = 0 # +1 per alignment if there's an exact match
    tp_lax = 0 # +1 per alignment if there's any overlap
    overlaps = []
    errors = []
    correct_nulls = 0

    for idx_align, alignment in enumerate(vr_rslts_lst):
        # skip alignments null on both sides
        if alignment == ([],[]):
            continue
        else:
            src_sents = alignment[0]
            tgt_sents = alignment[1]
            # get set of chapters from src, then from tgt
            chapters_from_src = set()
            chapters_from_tgt = set()
            # if alignment is null on src side, then chapters_from_src remains empty set
            if src_sents != []:
                for src_id in src_sents:
                    if isinstance(el_sent2section_dict[str(src_id)], list):
                        for section_name in el_sent2section_dict[str(src_id)]:
                            chapters_from_src.add(section_name)
                    else:
                        chapters_from_src.add(el_sent2section_dict[str(src_id)])
            # if alignment is null on tgt side, then chapters_from_tgt remains empty set
            if tgt_sents != []:
                for tgt_id in tgt_sents:
                    if isinstance(fr_sent2section_dict[str(tgt_id)], list):
                        for section_name_ in fr_sent2section_dict[str(tgt_id)]:
                            chapters_from_tgt.add(section_name_)
                    else:
                        chapters_from_tgt.add(fr_sent2section_dict[str(tgt_id)])

            # compare the sets, get tp strict and lax
            if chapters_from_src == chapters_from_tgt:
                tp_strict += 1

            # account for correct null : fr extraneous sections 
            elif chapters_from_src == set():
                tgt_counter = 0
                for chapter in chapters_from_tgt:
                    if chapter in fr_extra_section_names:
                        tgt_counter += 1
                # tp_strict if all tgt chapters are extraneous
                if tgt_counter == len(chapters_from_tgt):
                    tp_strict += 1
                    correct_nulls += 1

            else:
                overlap = chapters_from_src.intersection(chapters_from_tgt)
                if len(overlap) != 0:
                    tp_lax += 1
                    overlaps.append(alignment)
                else:
                    # save errors
                    error_dict = {}
                    error_dict["alignment"] = alignment
                    error_dict["alignmnent_idx"] = idx_align
                    error_dict["src_chapters"] = chapters_from_src
                    error_dict["tgt_chapters"] = chapters_from_tgt
                    errors.append(error_dict)
        
    return tp_strict, tp_lax, overlaps, errors, correct_nulls

################## Analyze vecalign results: by target sentences ######################
def build__src_2_tgt_dict(alignments_lst):
    '''
    If alignment is null on one side, inserts "null" 
    '''
    src_id_to_tgt_ids = defaultdict(set)
    for src, tgt in alignments_lst:
        if src == []:
            src = ["null"]
        if tgt == []:
            tgt = ["null"]
        for src_id in src:
            for tgt_id in tgt:
                src_id_to_tgt_ids[src_id].add(tgt_id)
    return src_id_to_tgt_ids

def build_tgt_2_src_dict(alignments_lst):
    '''
    If alignment is null on one side, inserts "null" 
    '''
    tgt_id_to_src_ids = defaultdict(set)
    for src, tgt in alignments_lst:
        if src == []:
            src = ["null"]
        if tgt == []:
            tgt = ["null"]
        for tgt_id in tgt:
            for src_id in src:
                tgt_id_to_src_ids[tgt_id].add(src_id)
    return tgt_id_to_src_ids

def score_fr_sents(fr2el_sent_aligns_dict, fr_sent2section_name_dict,
                   el_sent2section_name_dict, fr_extraneous_chapter_names):
    extraneous2null_tpstrict = 0
    extraneous2null_tplax = 0 # at least one overlap
    extraneous2text = 0 # no overlap

    text2text_tpstrict = 0
    text2text_tplax = 0
    text2text_incorrect = 0
    text2text_incorrect_lst = []

    text2null_incorrect = 0
    text2null_lst = []

    for fr_sent_idx in fr2el_sent_aligns_dict.keys():
    # for fr_sent_idx in [0,1000,10000]:
        # get grk sentences aligned to it
        el_aligned_sents = fr2el_sent_aligns_dict[fr_sent_idx]
        print(f"el aligned sents is {el_aligned_sents}")
        
        # TODO: necessary? to skip null-null alignments ("null" will not appear as key in dict)
        if str(fr_sent_idx) in fr_sent2section_name_dict.keys():
            # get fr sent chapter (keys are str). only 1 chapter per french sent
            fr_sent_chapter = list(fr_sent2section_name_dict[str(fr_sent_idx)])
            print(f"fr chapter is {fr_sent_chapter}")

            for tgt_chapter in fr_sent_chapter:
                if tgt_chapter in fr_extraneous_chapter_names:
                    # get num of fr - null alignments
                    extraneous2null_counter = 0
                    for item in el_aligned_sents:
                        if item == "null":
                            extraneous2null_counter += 1
                    # compare to number of el sents in alignmnent
                    if extraneous2null_counter == len(el_aligned_sents):
                        # then all grk aligned sents are null
                        extraneous2null_tpstrict += 1
                    elif extraneous2null_counter > 0:
                        # then at least one grk sent is null (also captures tpstrict)
                        extraneous2null_tplax += 1
                    else:
                        # no greek sents are null
                        extraneous2text += 1

                    # fr_extraneous2null_correct += el_counter/len(el_aligned_sents)
                    # fr_extraneous2text += (len(el_aligned_sents) - el_counter)/len(el_aligned_sents)
                        # if item == "null":
                        #     fr_extraneous2null_correct += 1
                        # else:
                        #     fr_extraneous2text += 1

                else: # compare fr and grk chapters
                    el_aligned_chapters = set()
                    el_text2text_correct_counter = 0
                    el_text2text_incorrect_counter = 0

                    for item in el_aligned_sents:
                        if item == "null":
                            text2null_incorrect += 1
                            text2null_lst.append(fr_sent_idx)
                        # if item == "null":
                        #     fr_text2null += 1
                        #     fr_text2null_lst.append(fr_sent_idx)
                        else:
                            # get chapters of el sent (keys are str)
                            if isinstance(el_sent2section_name_dict[str(item)], list):
                                for section_name in el_sent2section_name_dict[str(item)]:
                                    el_aligned_chapters.add(section_name)
                            else:
                                el_aligned_chapters.add(el_sent2section_name_dict[str(item)])

                    print(f"el chapters are {el_aligned_chapters}")

                    for item in el_aligned_chapters:
                        if tgt_chapter == item:
                            el_text2text_correct_counter += 1
                            # fr_text2text_correct += 1
                        else:
                            el_text2text_incorrect_counter += 1
                            # fr_text2text_incorrect += 1

                    if el_text2text_correct_counter == len(el_aligned_sents):
                        text2text_tpstrict += 1
                    elif el_text2text_correct_counter > 0:
                        text2text_tplax += 1
                    else:
                        text2text_incorrect += 1
                        text2text_incorrect_lst.append(fr_sent_idx)

                    # fr_text2text_correct += el_counter_text2text_correct/(len(el_aligned_sents))
                    # fr_text2text_incorrect += el_counter_text2text_incorrect/(len(el_aligned_sents))

    # remove text2null from text2text_incorrect_lst
    text2null_lst = set(text2null_lst)
    text2text_incorrect_lst = set(text2text_incorrect_lst)
    text2text_incorrect_lst -= text2null_lst
    # update num of text2text_incorrect
    text2text_incorrect -= text2null_incorrect
    
    results = [extraneous2null_tpstrict, extraneous2null_tplax, extraneous2text,
               text2text_tpstrict, text2text_tplax, 
               text2text_incorrect, text2text_incorrect_lst,
               text2null_incorrect, text2null_lst]
    
    return results