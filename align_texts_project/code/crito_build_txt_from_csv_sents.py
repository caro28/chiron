#!/usr/bin/env python

import re
import stanza
import argparse

import pandas as pd

from itertools import chain

from functions import (
    concatenate_txt, run_stanza, split_txt, flatten_list, write_file)


def segment_series(txt_series, lang):
    '''
    Modified from segment_en in preprocess_functions.py:
        uses split_txt function (works for en or el)
        removes text between % (Faroosh's comments)
    '''
    # join into one str
    series_str = concatenate_txt(txt_series)
    if lang == 'el':
        # split on ;:. for el
        return split_txt(series_str, lang)
    # TODO: update split_txt() for German and Persian
    else:
        # run stanza
        stanza_model = stanza.Pipeline(lang=lang, processors='tokenize')
        series_sents = run_stanza(series_str, stanza_model)
        # split further on ;:
        series_split = []
        for sent in series_sents:
            series_split.append(split_txt(sent, lang))
        return flatten_list(series_split)
    

def preprocess_series(txt_series, lang, keep_speaker_label, speaker_label_names):
    # convert all rows to string
    txt_series = txt_series.apply(str)
    
    # remove whitespace at beginning and end
    txt_series = txt_series.str.strip()

    # remove speaker labels if present
    # TODO: simpler way to do this?
    if keep_speaker_label == False:
        if lang == 'el':
            for idx, item in enumerate(txt_series):
                if item.startswith('Σωκράτης.'):
                    txt_series.loc[idx] = txt_series.loc[idx].lstrip('Σωκράτης.')
                elif item.startswith('Κρίτων.'):
                    txt_series.loc[idx] = txt_series.loc[idx].lstrip('Κρίτων.')
        else:
            for idx, item in enumerate(txt_series):
                if txt_series.loc[idx].startswith(speaker_label_names[0]):
                    txt_series.loc[idx] = txt_series.loc[idx].lstrip(speaker_label_names[0])
                elif txt_series.loc[idx].startswith(speaker_label_names[1]):
                    txt_series.loc[idx] = txt_series.loc[idx].lstrip(speaker_label_names[1])

        # remove whitespace at beginning and end
        txt_series = txt_series.str.strip()

    # split text into sentences
    series_split = segment_series(txt_series, lang)
    
    # save as df and change col name
    series_df = pd.DataFrame(series_split)
    series_df.columns = ['text']
    
    # remove whitespace at beginning and end
    series_df['text'] = series_df['text'].str.strip()

    # drop rows with NaN
    series_df.dropna(how='any', inplace=True)

    # # drop rows with empty strings
    # series_df.drop(series_df.loc[series_df['text']==''].index, inplace=True)

    # send to list
    series_lst = list(series_df['text'])

    return series_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess series in raw data csv and write to txt files")

    parser.add_argument("--csv_path", type=str, required=True,
    help="Path to raw data csv")

    parser.add_argument("--dev_range", type=int, nargs=2, 
    help="Index of first and last row of dev set")

    parser.add_argument("--test_range", type=int, nargs=2, 
    help="Index of first and last row of test set")

    parser.add_argument("--series_lang", type=str, nargs=2, required=True, 
    help="Source lang first, then Tgt lang: el for Greek, en for English, de for German, pers for Persian")

    parser.add_argument("--series_cols", type=str, nargs=2, required=True, 
    help="Source first, then Tgt. Use column names from csv")

    parser.add_argument("--speaker_label", type=bool, default=True,
    help="For Crito, whether to remove speaker label in dialogue")

    parser.add_argument("--speaker_labels_tgt", type=str, nargs=2, default=None, 
    help="For Crito, if removing speaker labels, then list labels for Socrates then Crito in tgt language")

    parser.add_argument("--fileout_names_src", type=str, nargs=2, required=True, 
    help="Path out including file names: src dev, then src test")

    parser.add_argument("--fileout_names_tgt", type=str, nargs=2, required=True, 
    help="Path out including file names: tgt dev, then tgt test")

    args = parser.parse_args()

    print("loading raw data csv")
    # load raw data csv
    path_in = args.csv_path
    df = pd.read_csv(path_in)

    print("building dev and test sets")
    # get dev_df and test_df
    if args.dev_range is not None:
        dev_df = df.iloc[args.dev_range[0]:args.dev_range[1]]
        test_df = df.iloc[args.test_range[0]:args.test_range[1]]
    else:
        # default: dev is first 80% of csv
        cut_off = round(len(df)*.8)
        dev_df = df.iloc[:cut_off]
        test_df = df.iloc[cut_off:]

    dev_df = dev_df.fillna("")
    test_df = test_df.fillna("")

    print("getting dev and test series")
    # get dev and test series
    src_col = str(args.series_cols[0])
    tgt_col = str(args.series_cols[1])
    src_series = [dev_df[src_col], test_df[src_col]]
    tgt_series = [dev_df[tgt_col], test_df[tgt_col]]

    print("getting paths out")
    # pathsout: each contains list of 2 items (dev then tst path+filename)
    src_pathsout = args.fileout_names_src
    tgt_pathsout = args.fileout_names_tgt
    
    print("preprocessing series")
    # preprocess series and write to file
    src_lang = args.series_lang[0]
    tgt_lang = args.series_lang[1]
    for idx, series in enumerate(src_series):
        processed_txt = preprocess_series(src_series[idx], src_lang, args.speaker_label, args.speaker_labels_tgt)
        print("processed a series")
        path = src_pathsout[idx]
        write_file(processed_txt, path)
        print("wrote text to file")

    for idx, series in enumerate(tgt_series):
        print("processed a series")
        processed_txt = preprocess_series(tgt_series[idx], tgt_lang, args.speaker_label, args.speaker_labels_tgt)
        path = tgt_pathsout[idx]
        write_file(processed_txt, path)
        print("wrote text to file")

