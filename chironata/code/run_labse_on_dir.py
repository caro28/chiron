import os, glob
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

from functions import load_txt_as_lst

#===================== FUNCTIONS =======================#
def build_embeddings_huggingface(sentences_lst, model_):
    embeddings = model_.encode(sentences_lst)
    return embeddings

def write_to_file(embeds_lst, file_out):
    embeds_lst = np.array(embeds_lst)
    with open(file_out, 'w') as f:
        embeds_lst.tofile(f)
# =========================================================#

labse_vec_gpu = 'srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate labse_vec_pipeline; {command};"'

sents_dir = sys.argv[1]
print(f"working on dir {sents_dir}")
dir_out = sys.argv[2]

for sents in glob.iglob(sents_dir+"*.sents"):
    if os.path.isfile(dir_out+".emb") != False:

    # ctsurn = os.path.splitext(os.path.basename(src_emb))[0]
    # src_prefix = src_dir+ctsurn
    # tgt_files = lookup_table[ctsurn]
    # for tgt_name in tgt_files:
    #     tgt_prefix = tgt_dir+tgt_name
    #     # check if a fr translation exists
    #     if os.path.isfile(tgt_prefix+".emb") != False: 
    #         params['command'] = f'./vecalign.py --alignment_max_size 8 --src {src_prefix+".sents"} --tgt {tgt_prefix+".sents"} --src_embed {src_prefix+".overlaps"} {src_emb} --tgt_embed {tgt_prefix+".overlaps"} {tgt_prefix+".emb"}'
    #         # print(labse_vec_cpu.format(**params))
    #         if os.path.isfile(f"{rslts_dir}{ctsurn}_{tgt_name}.rslts") == False:
    #             print(f"working on {ctsurn}_{tgt_name}")
    #             subprocess.run(labse_vec_cpu.format(**params),shell=True)
    
