#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd

from functions import write_file

def check_empty_rows(text_series):
    idx_delete = []
    for idx, chunk in enumerate(text_series):
        if chunk == "":
            idx_delete.append(idx)
    return idx_delete
    

def delete_empty_chunks(text_series, list_idx):
    for idx in sorted(list_idx, reverse=True):
        del text_series[idx]


def write_to_file(file_name, txt_lst):
    with open(file_name, 'w') as f:
        for chunk in txt_lst:
            f.write(f"{chunk}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text chunks (one per row) from raw data csv, preprocess, then write to file (for embedding)")

    parser.add_argument("--csv_path", type=str, required=True,
    help="Path to raw data csv")

    parser.add_argument("--series_cols", type=str, nargs=2, required=True, 
    help="Source first, then Tgt. Use column names from csv")

    parser.add_argument("--fileout_src", type=str, required=True, 
    help="Path out including file name")

    parser.add_argument("--fileout_tgt", type=str, required=True, 
    help="Path out including file name")

    args = parser.parse_args()

    print("loading raw data csv")
    # load raw data csv
    path_in = args.csv_path
    df = pd.read_csv(path_in)

    # fill NAN with empty strings
    df = df.fillna("")

    # get src and tgt series
    src_col = str(args.series_cols[0])
    tgt_col = str(args.series_cols[1])
    print(f"src column name is {src_col} and tgt is {tgt_col}")

    # convert to list for indexing
    src_chunks = list(df[src_col])
    tgt_chunks = list(df[tgt_col])

    # get empty rows
    tgt_empty_lst = check_empty_rows(tgt_chunks)

    # delete chunk in greek and tgt
    delete_empty_chunks(src_chunks, tgt_empty_lst)
    delete_empty_chunks(tgt_chunks, tgt_empty_lst)

    print("getting paths out")
    src_pathout = args.fileout_src
    tgt_pathout = args.fileout_tgt

    write_file(src_chunks, src_pathout)
    write_file(tgt_chunks, tgt_pathout)
