#!/usr/bin/env python

import os, sys, glob
import lxml
import subprocess
from tqdm import tqdm

'''
Data processing pipeline.
If running on directory of Ancient Greek or Latin texts, lang = "src"
args in this order:
src_dir: data files to process
lang: src, fr, en, it, de
spacy_model_: fr_core_news_sm, en_core_web_sm, it_core_news_sm, de_core_news_sm
'''

params = {
    'QUEUE':'short',
    'SWORKERS':2,
    'SMEM':10
}

# slurm_run = 'srun --time 1-0 -p {QUEUE} -N 1 -c {SWORKERS} --mem={SMEM}G bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate use_lxml; {command};"'
lxml_cpu = 'srun --time 1-0 --partition=short --nodes=1 --pty --mem=4G --time=00:30:00 /bin/bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate use_lxml; {command};"'
labse_vec_cpu = 'srun --partition=short --nodes=1 --pty --mem=4GB --time=01:00:00 /bin/bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate labse_vec_pipeline; {command};"'
labse_vec_gpu = 'srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-sxm2 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate labse_vec_pipeline; {command};"'

src_dir = sys.argv[1]
lang = sys.argv[2]
spacy_model_ = sys.argv[3]

if lang == "src":
    for path in tqdm(glob.iglob(src_dir+"*.txt")):
        prefix = os.path.splitext(path)[0]
        
        # # Step 1: Sentence Segmentation
        if os.path.isfile(prefix+".sents") == False:
            params['command'] = f'./segment_sents.py {path} {lang}'
            print(f'started a run on file {prefix}')
            subprocess.run(labse_vec_gpu.format(**params),shell=True,check=True)
        
        # # Step 2: Overlap builder
        if os.path.isfile(prefix+".overlaps") == False:
            params['command'] = f'./overlap.py {prefix+".sents"}'
            print("building overlaps")
            subprocess.run(labse_vec_cpu.format(**params), shell=True,check=True)
        
        # # Step 3: Embedder
        if os.path.isfile(prefix+".emb") == False:
            params['command'] = f'./run_labse.py {prefix+".overlaps"}'
            print('labse run')
            subprocess.run(labse_vec_gpu.format(**params), shell=True,check=True)

else:
    for path in tqdm(glob.iglob(src_dir+"*.xml")):
        prefix = os.path.splitext(path)[0]

        # Step 1: Extract from XML
        if os.path.isfile(prefix+".par") == False:
            params['command'] = f'./book-stream.py {path}'
            print(f"starting on new file {prefix}")
            subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

        # Step 2: Clean XML output
        if os.path.isfile(prefix+".txt") == False:
            params['command'] = f'./clean_par.py {prefix+".par"} {spacy_model_} {lang}'
            print("cleaning pars")
            subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

        # Step 3: Sentence Segmentation
        if os.path.isfile(prefix+".sents") == False:
            params['command'] = f'./segment_sents.py {prefix+".txt"} {lang}'
            print("splitting sents")
            subprocess.run(labse_vec_gpu.format(**params),shell=True,check=True)

        # Step 2: Overlap builder
        if os.path.isfile(prefix+".overlaps") == False:
            params['command'] = f'./overlap.py {prefix+".sents"}'
            print("building overlaps")
            subprocess.run(labse_vec_cpu.format(**params), shell=True,check=True)
        
        #Step 3: Embedder
        if os.path.isfile(prefix+".emb") == False:
            params['command'] = f'./run_labse.py {prefix+".overlaps"}'
            print('labse run')
            subprocess.run(labse_vec_gpu.format(**params), shell=True,check=True)`





# for path in tqdm(glob.iglob("/scratch/craig.car/src_data/*.txt")):
#     prefix = os.path.splitext(path)[0]
#     lang = "src"
    
#     # # Step 1: Sentence Segmentation
#     if os.path.isfile(prefix+".sents") == False:
#         params['command'] = f'./segment_sents.py {path} {lang}'
#         print(f'started a run on file {prefix}')
#         subprocess.run(labse_vec_gpu.format(**params),shell=True,check=True)
    
#     # # Step 2: Overlap builder
#     if os.path.isfile(prefix+".overlaps") == False:
#         params['command'] = f'./overlap.py {prefix+".sents"}'
#         print("building overlaps")
#         subprocess.run(labse_vec_cpu.format(**params), shell=True,check=True)
    
#     # # Step 3: Embedder
#     if os.path.isfile(prefix+".emb") == False:
#         params['command'] = f'./run_labse.py {prefix+".overlaps"}'
#         print('labse run')
#         subprocess.run(labse_vec_gpu.format(**params), shell=True,check=True)
        
# for path in tqdm(glob.iglob("/scratch/craig.car/french_trans-dev/*.xml")):
# for path in tqdm(glob.iglob("/scratch/craig.car/french_trans-dev/as_yet_unedited_DDD/*.xml")):
#     prefix = os.path.splitext(path)[0]
#     lang = "fr"
#     spacy_model_ = "fr_core_news_sm"

#     # Step 1: Extract from XML
#     if os.path.isfile(prefix+".par") == False:
#         params['command'] = f'./book-stream.py {path}'
#         print(f"starting on new file {prefix}")
#         print("ran on fr")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 2: Clean XML output
#     if os.path.isfile(prefix+".txt") == False:
#         params['command'] = f'./clean_par.py {prefix+".par"} {spacy_model_} {lang}'
#         print("cleaning pars")
#         print("ran on fr")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 3: Sentence Segmentation
#     if os.path.isfile(prefix+".sents") == False:
#         params['command'] = f'./segment_sents.py {prefix+".txt"} {lang}'
#         print("splitting sents")
#         print("ran on fr")
#         subprocess.run(labse_vec_gpu.format(**params),shell=True,check=True)

#     # Step 2: Overlap builder
#     if os.path.isfile(prefix+".overlaps") == False:
#         params['command'] = f'./overlap.py {prefix+".sents"}'
#         print("building overlaps")
#         print("ran on fr")
#         subprocess.run(labse_vec_cpu.format(**params), shell=True,check=True)
    
#     #Step 3: Embedder
#     if os.path.isfile(prefix+".emb") == False:
#         params['command'] = f'./run_labse.py {prefix+".overlaps"}'
#         print('labse run')
#         print("ran on fr")
#         subprocess.run(labse_vec_gpu.format(**params), shell=True,check=True)
 

# for path in tqdm(glob.iglob("/scratch/craig.car/english_trans-dev/*.xml")):
#     prefix = os.path.splitext(path)[0]
#     lang = "en"
#     spacy_model_ = "en_core_web_sm"

#     # Step 1: Extract from XML
#     if os.path.isfile(prefix+".par") == False:
#         params['command'] = f'./book-stream.py {path}'
#         print(f"starting on new file {prefix}")
#         print("ran on en")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 2: Clean XML output
#     if os.path.isfile(prefix+".txt") == False:
#         params['command'] = f'./clean_par.py {prefix+".par"} {spacy_model_} {lang}'
#         print("cleaning pars")
#         print("ran on en")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 3: Sentence Segmentation
#     if os.path.isfile(prefix+".sents") == False:
#         params['command'] = f'./segment_sents.py {prefix+".txt"} {lang}'
#         print("splitting sents")
#         print("ran on en")
#         subprocess.run(labse_vec_gpu.format(**params),shell=True,check=True)

#     # Step 2: Overlap builder
#     if os.path.isfile(prefix+".overlaps") == False:
#         params['command'] = f'./overlap.py {prefix+".sents"}'
#         print("building overlaps")
#         print("ran on en")
#         subprocess.run(labse_vec_cpu.format(**params), shell=True,check=True)
    
#     #Step 3: Embedder
#     if os.path.isfile(prefix+".emb") == False:
#         params['command'] = f'./run_labse.py {prefix+".overlaps"}'
#         print('labse run')
#         print("ran on en")
#         subprocess.run(labse_vec_gpu.format(**params), shell=True,check=True)
 
# for path in tqdm(glob.iglob("/scratch/craig.car/italian_trans-dev/as_yet_unedited_DDD/*.xml")):
# for path in tqdm(glob.iglob("/scratch/craig.car/italian_trans-dev/*.xml")):
#     prefix = os.path.splitext(path)[0]
#     lang = "it"
#     spacy_model_ = "it_core_news_sm"

#     # Step 1: Extract from XML
#     if os.path.isfile(prefix+".par") == False:
#         params['command'] = f'./book-stream.py {path}'
#         print(f"starting on new file {prefix}")
#         print("ran on it")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 2: Clean XML output
#     if os.path.isfile(prefix+".txt") == False:
#         params['command'] = f'./clean_par.py {prefix+".par"} {spacy_model_} {lang}'
#         print("cleaning pars")
#         print("ran on it")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 3: Sentence Segmentation
#     if os.path.isfile(prefix+".sents") == False:
#         params['command'] = f'./segment_sents.py {prefix+".txt"} {lang}'
#         print("splitting sents")
#         print("ran on it")
#         subprocess.run(labse_vec_gpu.format(**params),shell=True,check=True)

#     # Step 2: Overlap builder
#     if os.path.isfile(prefix+".overlaps") == False:
#         params['command'] = f'./overlap.py {prefix+".sents"}'
#         print("building overlaps")
#         print("ran on it")
#         subprocess.run(labse_vec_cpu.format(**params), shell=True,check=True)
    
#     #Step 3: Embedder
#     if os.path.isfile(prefix+".emb") == False:
#         params['command'] = f'./run_labse.py {prefix+".overlaps"}'
#         print('labse run')
#         print("ran on it")
#         subprocess.run(labse_vec_gpu.format(**params), shell=True,check=True)


# for path in tqdm(glob.iglob("/scratch/craig.car/german_trans-dev/*.xml")):
#     prefix = os.path.splitext(path)[0]
#     lang = "de"
#     spacy_model_ = "de_core_news_sm"

#     # Step 1: Extract from XML
#     if os.path.isfile(prefix+".par") == False:
#         params['command'] = f'./book-stream.py {path}'
#         print(f"starting on new file {prefix}")
#         print("ran on de")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 2: Clean XML output
#     if os.path.isfile(prefix+".txt") == False:
#         params['command'] = f'./clean_par.py {prefix+".par"} {spacy_model_} {lang}'
#         print("cleaning pars")
#         print("ran on de")
#         subprocess.run(lxml_cpu.format(**params),shell=True,check=True)

#     # Step 3: Sentence Segmentation
#     if os.path.isfile(prefix+".sents") == False:
#         params['command'] = f'./segment_sents.py {prefix+".txt"} {lang}'
#         print("splitting sents")
#         print("ran on de")
#         subprocess.run(labse_vec_gpu.format(**params),shell=True,check=True)

#     # Step 2: Overlap builder
#     if os.path.isfile(prefix+".overlaps") == False:
#         params['command'] = f'./overlap.py {prefix+".sents"}'
#         print("building overlaps")
#         print("ran on de")
#         subprocess.run(labse_vec_cpu.format(**params), shell=True,check=True)
    
#     #Step 3: Embedder
#     if os.path.isfile(prefix+".emb") == False:
#         params['command'] = f'./run_labse.py {prefix+".overlaps"}'
#         print('labse run')
#         print("ran on de")
#         subprocess.run(labse_vec_gpu.format(**params), shell=True,check=True)
