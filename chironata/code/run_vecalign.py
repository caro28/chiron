import glob, os
import subprocess
import json
import sys

params = {
    'QUEUE':'short',
    'SWORKERS':2,
    'SMEM':4
}

labse_vec_cpu = 'srun --partition=short --nodes=1 --mem=4GB --time=01:00:00 /bin/bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate labse_vec_pipeline; {command};"'

src_dir = "/scratch/craig.car/src_data/"
tgt_dir = sys.argv[1]
print(tgt_dir)

rslts_dir = "/home/craig.car/repos/chiron/chironata/data/alignments_rslts/"

lookup_table_path = "/home/craig.car/repos/chiron/chironata/data/cts_lookup_table.json"
with open(lookup_table_path) as f:
    lookup_table = json.load(f)

for src_emb in glob.iglob(src_dir+"*.emb"):
    ctsurn = os.path.splitext(os.path.basename(src_emb))[0]
    src_prefix = src_dir+ctsurn
    tgt_files = lookup_table[ctsurn]
    for tgt_name in tgt_files:
        # if tgt_name.endswith(".xml"):
        #     tgt_name = tgt_name.split(".xml")[0]
        #     print(tgt_name)
        tgt_prefix = tgt_dir+tgt_name
        # check if an emb file exists for this translation exists
        if os.path.isfile(tgt_prefix+".emb") != False: 
            params['command'] = f'./vecalign.py --alignment_max_size 8 --src {src_prefix+".sents"} --tgt {tgt_prefix+".sents"} --src_embed {src_prefix+".overlaps"} {src_emb} --tgt_embed {tgt_prefix+".overlaps"} {tgt_prefix+".emb"}'
            # print(labse_vec_cpu.format(**params))
            if os.path.isfile(f"{rslts_dir}{ctsurn}_{tgt_name}.rslts") == False:
                print(f"working on {ctsurn}_{tgt_name}")
                subprocess.run(labse_vec_cpu.format(**params),shell=True)
    
