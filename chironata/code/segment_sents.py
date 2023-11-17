#!/usr/bin/env python

import os, sys
import stanza

import pandas as pd

from itertools import chain

from functions import (
    load_txt_as_lst, split_txt, flatten_list, write_file)

def run_stanza(text_str, model_):
    '''
    returns sentences as list
    '''
    print("running stanza")
    doc = model_(text_str)
    return [sentence.text for sentence in doc.sentences]


def segment_series(txt_str, lang, model):
    if lang == 'el':
        # split on ;:. for el
        return split_txt(txt_str, lang)
    else:
        # run stanza
        series_sents = run_stanza(txt_str, model)
        # split further on ;:
        series_split = []
        for sent in series_sents:
            series_split.append(split_txt(sent, lang))
        return flatten_list(series_split)
    

def preprocess_series(txt_lst, lang, model):
    # remove whitespace at beginning and end
    cleaned_str = ""
    # join into str
    for row in txt_lst:
        if row == "":
            continue
        else:
            cleaned_str += row.strip()
   
    # split text into sentences
    series_split = segment_series(cleaned_str, lang, model)
    
    # save as df and change col name
    series_df = pd.DataFrame(series_split)
    series_df.columns = ['text']
    
    # remove whitespace at beginning and end
    series_df['text'] = series_df['text'].str.strip()

    # drop rows with NaN
    series_df.dropna(how='any', inplace=True)

    # send to list
    series_lst = list(series_df['text'])

    return series_lst


if __name__ == "__main__":
    # load file
    file = sys.argv[1]
    # convert to list (src files will have length 1)
    text_lst = load_txt_as_lst(file)

    # set lang for files from src_data dir
    if "greek" in sys.argv[1]:
        lang = "el"
        stanza_model = ""
    elif "latin" in sys.argv[1]:
        lang = "la"
    else:
        lang = sys.argv[2]
    # load stanza model
    stanza_model = stanza.Pipeline(lang=lang, processors='tokenize', use_gpu=True)
    
    print(f"lang is {lang}")

    # preprocess text
    text_sents = preprocess_series(text_lst, lang, stanza_model)
    
    # write to file as .sents
    prefix = os.path.splitext(file)[0]
    path_out = prefix+".sents"
    write_file(text_sents, path_out)