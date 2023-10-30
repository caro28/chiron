QUEUE=short
SWORKERS=2
SMEM=10
slurm_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate use_lxml; $(1)";

labse_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate labse_vec_pipeline; $(1)";

# SRC=french_trans-dev german_trans-dev italian_trans-dev
SRC_trans=french_trans-dev
SRC_src=source_texts

.SECONDARY:

proc.done:	$(patsubst %,data/french_trans-dev/%,$(patsubst %.xml,%_emb.dummy,$(shell find $(SRC_trans) -name '*.xml')))
	touch $@

# src.done:	$(patsubst %,data/%,$(patsubst %.txt,%_emb.dummy,$(shell find $(SRC_trans) -name '*.txt')))
# 	touch $@

data/french_trans-dev/%._par:	%.xml
	$(call slurm_run,./book-stream.py $< > $@)

data/french_trans-dev/%.txt:	data/french_trans-dev/%.par
	$(call slurm_run,./run_spacy.py $< > $@)

data/french_trans-dev/%.sents:	data/french_trans-dev/%.txt
	$(call slurm_run,./segment_sents.py $< > $@)

data/french_trans-dev/%.overlaps:	data/french_trans-dev/%.sents
	$(call labse_run,./overlap.py $< > $@)

data/french_trans-dev/%_emb.dummy:	data/french_trans-dev/%.overlaps
	$(call labse_run,./run_labse.py $< > $@)



data/source_texts/%.sents:	data/source_texts/%.txt
	$(call slurm_run,./segment_sents.py $< > $@)

data/source_texts/%.overlaps:	data/source_texts/%.sents
	$(call labse_run,./overlap.py $< > $@)

data/source_texts/%_emb.dummy:	data/source_texts/%.overlaps
	$(call labse_run,./run_labse.py $< > $@)