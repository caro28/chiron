#!/usr/bin/env python

import os, sys, glob
import subprocess
from tqdm import tqdm

params = {
    'QUEUE':'short',
    'SWORKERS':2,
    'SMEM':10
}

slurm_run = 'srun --time 1-0 -p {QUEUE} -N 1 -c {SWORKERS} --mem={SMEM}G bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate use_lxml; {command};"'
labse_run = 'srun --time 1-0 -p {QUEUE} -N 1 -c {SWORKERS} --mem={SMEM}G bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate labse_vec_pipeline; {command};"'

# for path in tqdm(glob.iglob("/scratch/craig.car/src_data/*.txt")):
#     prefix = os.path.splitext(path)[0]
#     lang = "src"
    
#     # Step 1: Sentence Segmentation
#     params['command'] = f'./segment_sents.py {path} {lang}'
#     subprocess.run(slurm_run.format(**params),shell=True)
    
#     # Step 2: Overlap builder
#     params['command'] = f'./overlap.py {prefix+".sents"}'
#     subprocess.run(labse_run.format(**params), shell=True)
    
#     # Step 3: Embedder
#     params['command'] = f'./run_labse.py {prefix+".overlaps"}'
#     subprocess.run(labse_run.format(**params), shell=True)
    
#     break

for path in tqdm(glob.iglob("/scratch/craig.car/french_trans-dev/*.xml")):
    prefix = os.path.splitext(path)[0]
    lang = "fr"

    # # Step 1: Extract from XML
    # params['command'] = f'./book-stream.py {path}'
    # subprocess.run(slurm_run.format(**params),shell=True)

    # # Step 2: Clean XML output
    # spacy_model_ = "fr_core_news_sm"
    # params['command'] = f'./clean_par.py {prefix+".par"} {spacy_model_} {lang}'
    # subprocess.run(slurm_run.format(**params),shell=True)

    # # Step 3: Sentence Segmentation
    # params['command'] = f'./segment_sents.py {prefix+".txt"} {lang}'
    # subprocess.run(slurm_run.format(**params),shell=True)

    # # Step 2: Overlap builder
    # params['command'] = f'./overlap.py {prefix+".sents"}'
    # subprocess.run(labse_run.format(**params), shell=True)
    
    #Step 3: Embedder
    params['command'] = f'./run_labse.py {prefix+".overlaps"}'
    subprocess.run(labse_run.format(**params), shell=True)

    break 