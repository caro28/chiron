#!/usr/bin/env python

import sys, os
import json
import spacy
import spacy_fastlang

from functions import load_txt_as_lst, write_file

def clean_par(pars_lst, spacy_model, lang):
    '''
    1. Merges sections split during extraction from XML while filtering out
    text in different language (e.g. for editions including original text).
    2. Builds metadata in the form of a dict mapping the text's index to
    section name. 
    '''
    reconstructed = []
    section_idx2section_name = {}
    prev_meta = pars_lst[0].split("\t")[0]
    current_txt = ""
    reconstructed_idx = 0
    for idx, fragment in enumerate(pars_lst):
        meta = fragment.split("\t")[0]
        text = fragment.split("\t")[2]
        if meta == prev_meta:
            # still in same section, check lang before adding text
            if spacy_model(text.lower())._.language == lang:
                current_txt += " "+text
        else:
            # started a new section, append current_txt
            if current_txt != "":
                reconstructed.append(current_txt)
                # add metadata to dict
                section_idx2section_name[reconstructed_idx] = prev_meta
                # re-initialize reconstructed_idx regardless of lang
                reconstructed_idx += 1
            # re-initialize prev_meta
            prev_meta = meta
            # check lang then re-initialize current_txt
            if spacy_model(text.lower())._.language == lang:
                current_txt = text
            else:
                current_txt = ""
    # add last section to reconstructed
    reconstructed.append(current_txt)
    section_idx2section_name[reconstructed_idx] = meta    
    return reconstructed, section_idx2section_name


if __name__ == '__main__':
    file = sys.argv[1]
    spacy_model_ = spacy.load(str(sys.argv[2]))
    spacy_model_.add_pipe("language_detector")

    text_par = load_txt_as_lst(file)

    # check lang, reconstruct pars, and get metadata
    lang_ = sys.argv[3]
    clean_pars, idx2section_name = clean_par(text_par, spacy_model_, lang_)

    # write both to file
    prefix = os.path.splitext(file)[0]
    # check for empty file
    if len(clean_pars) != 0:
        write_file(clean_pars, prefix+".txt")
        with open(prefix+".json", 'w') as fp:
            json.dump(idx2section_name, fp)
    else:
        raise SystemExit(f"empty file after clean_par.py for {prefix}")
