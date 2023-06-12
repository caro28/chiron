def load_txt_as_lst(path_in):
    output_lst = []
    with open(path_in, "rt") as f:
        for line in f:
            output_lst.append(line)
    return output_lst
