import re
import pandas as pd
from itertools import chain

def load_txt_as_lst(path_in):
    output_lst = []
    with open(path_in, "rt") as f:
        for line in f:
            output_lst.append(line)
    return output_lst

def split_txt(txt_str, lang):
    if lang=='el':
        split_lst = re.split("([;:.])", txt_str)
        delimiters = ";:."
    else:
        # split keeping split char
        split_lst = re.split("([;:])", txt_str)
        delimiters = ";:"        
    # add delimiters back to previous token
    delete_idx = []
    for idx, phrase in enumerate(split_lst):
        if phrase in delimiters:
            split_lst[idx-1] = split_lst[idx-1]+split_lst[idx]
            delete_idx.append(idx)
    for index in sorted(delete_idx, reverse=True):
        del split_lst[index]
    split_lst_no_newlines = []
    for idx, phrase in enumerate(split_lst):
        split_lst_no_newlines.extend(phrase.split("\n"))
    # return split_lst
    return split_lst_no_newlines

def run_stanza(text_str, model_):
    '''
    returns sentences as list
    '''
    doc = model_(text_str)
    return [sentence.text for sentence in doc.sentences]

def flatten_list(nested_list):
    return list(chain.from_iterable(nested_list))

def segment_series(txt_str, lang, stanza_model):
    if lang == 'el':
        # split on ;:. for el
        return split_txt(txt_str, lang)
    else:
        # run stanza
        series_sents = run_stanza(txt_str, stanza_model)
        # split further on ;:
        series_split = []
        for sent_idx, sent in enumerate(series_sents):
            new_sent = split_txt(sent, lang)
            series_split.append(new_sent)
        return flatten_list(series_split)

def preprocess_series(txt_str, lang, stanza_model):
    # split text into sentences
    series_split = segment_series(txt_str, lang, stanza_model)
    print("segmented str into sentences")
    # save as df and change col name
    series_df = pd.DataFrame(series_split)
    series_df.columns = ['text']
    # remove whitespace at beginning and end
    series_df['text'] = series_df['text'].str.strip()
    # drop rows with NaN
    series_df.dropna(how='any', inplace=True)
    # drop rows with empty strings
    series_df.drop(series_df.loc[series_df['text']==''].index, inplace=True)
    # send to list
    series_lst = list(series_df['text'])
    return series_lst

def write_file(input_lst, name_out):
    filename = name_out
    with open(filename, 'w') as file:
        for sentence in input_lst:
            file.write(f"{sentence}\n")