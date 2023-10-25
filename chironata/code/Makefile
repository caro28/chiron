QUEUE=short
SWORKERS=2
SMEM=10
slurm_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate use_lxml; $(1)";

labse_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "source /home/craig.car/miniconda3/bin/activate; conda activate labse_vec_pipeline; $(1)";

# SRC=french_trans-dev german_trans-dev italian_trans-dev
SRC=french_trans-dev

.SECONDARY:

proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.emb,$(shell find $(SRC) -name '*.xml')))
	touch $@

proc/%.par:	%.xml
	mkdir -p $(dir $@)
	$(call slurm_run,./book-stream.py $< > $@)

proc/french_trans-dev/%.txt:	proc/french_trans-dev/%.par
	$(call slurm_run,./run_spacy.py $< > $@)

proc/french_trans-dev/%_sents.txt:	proc/french_trans-dev/%.txt
	$(call slurm_run,./segment_sents.py $< > $@)

proc/french_trans-dev/%_overlaps.fr:	proc/french_trans-dev/%_sents.txt
	$(call labse_run,./overlap.py $< > $@)

proc/french_trans-dev/%.emb:	proc/french_trans-dev/%_overlaps.fr
	$(call labse_run,./run_labse.py $< > $@)