import re
import pandas as pd
from itertools import chain
from ast import literal_eval
from collections import defaultdict

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

def build_sent_to_section_dict(lst_tokenized_sents, lst_tokenized_chapts,
                               dict_chapter_2_section):
    """
    Build dict of sentence idx to section name
    """
    sent_idx_2_section_name = {}
    token_counter = 0 # per section/chapter
    current_section_idx = 0
    for idx_sent, sent in enumerate(lst_tokenized_sents):
        token_counter += len(sent)
        current_chapter_length = len(lst_tokenized_chapts[current_section_idx])
        if token_counter < current_chapter_length:
            # add sent to dict
            sent_idx_2_section_name[idx_sent] = dict_chapter_2_section[current_section_idx]
        elif token_counter == current_chapter_length:
            # add sent to dict as part of current section
            sent_idx_2_section_name[idx_sent] = dict_chapter_2_section[current_section_idx]
            # reset token counter and current section idx for next sent iteration
            token_counter = 0
            current_section_idx += 1
        else: # token_counter > current_chapter_length, i.e. we've crossed a section boundary 
            # add sent to current section and next section
            sent_idx_2_section_name[idx_sent] = [
                dict_chapter_2_section[current_section_idx], 
                dict_chapter_2_section[current_section_idx+1]
            ]
            # adjust token counter by only including portion of sent in new section
            token_counter = token_counter - current_chapter_length
            # update current section idx for next sent iteration
            current_section_idx += 1
    return sent_idx_2_section_name

def concatenate_txt(txt_series):
    '''
    Converts to str (in case of NaN present as float) and concatenates rows 
    into one continuous string
    '''
    # convert all rows to string
    txt_series = txt_series.apply(str)
    # join into a single string
    return ' '.join(txt_series)

def get_perseus_txt_by_book(df, cts_tag, num_books):
    '''
    Extract Perseus text in df by book
    '''
    txt_by_book = []
    idx2book_name = {}
    idx_counter = 0
    for book_idx in range(1, num_books+1):
        loc_tag = cts_tag + str(book_idx)
        book_text = concatenate_txt(df[df['loc'].str.startswith(loc_tag)]['text'].replace('\n',' ', regex=True))
        txt_by_book.append(book_text)
        # add to dict. chap name format: "booknum"
        book_name = str(book_idx)
        idx2book_name[idx_counter] = book_name
        idx_counter += 1
    return txt_by_book, idx2book_name

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