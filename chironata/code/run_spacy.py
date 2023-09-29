#!/usr/bin/env python

import sys
import json
import glob
import spacy
import spacy_fastlang

'''
For files in french_trans-dev only
TODO: how to expand to other langs? Need to load lang model + specify lang to check against
'''

def load_txt_as_lst(path_in):
    output_lst = []
    with open(path_in, "rt") as f:
        for line in f:
            output_lst.append(line)
    return output_lst

def apply_spacy(model, input_par, lang):
    kept_text = []
    kept_scores = []
    excluded_text = []

    for idx, item in enumerate(input_par):
        if model(item)._.language == lang:
            kept_text.append(item)
            kept_scores.append(model(item)._.language_score)
        else:
            excluded_text.append(item)
        # print(f"processed row {idx}")
    return kept_text, kept_scores, excluded_text

def clean_kept(kept_text):
    idx2metadata = {}
    text_clean = []
    for idx, item in enumerate(kept_text):
        idx_dict = {}
        split = item.split("\t")
        idx_dict["section_name"] = split[0]
        idx_dict["page_num"] = split[1]
        idx2metadata[idx] = idx_dict
        text_clean.append(split[2])

    return text_clean, idx2metadata

def write_file(input_lst, name_out):
    filename = name_out
    with open(filename, 'w') as file:
        for sentence in input_lst:
            file.write(f"{sentence}\n")

if __name__ == '__main__':
    file = sys.argv[1]
    spacy_fr_sm = spacy.load("fr_core_news_sm")
    spacy_fr_sm.add_pipe("language_detector")

    text_par = load_txt_as_lst(file)

    # run spacy and extract kept text, scores per par, and excluded text
    lang_ = "fr"
    lang_text, par_scores, excl_text = apply_spacy(spacy_fr_sm, text_par, lang_)

    # get clean text and metadata dicts
    clean_text_, idx2metadata_ = clean_kept(lang_text)

    # print clean text to stdout
    for sentence in clean_text_:
        print(sentence)

    # save metadata dict to file
    path_out_dict = "/home/craig.car/repos/chiron/chironata/code/proc/french_trans-dev/"+file[40:-4]+"_metadata.json"
    with open(path_out_dict, 'w') as fp:
        json.dump(idx2metadata_, fp)



