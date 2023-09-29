QUEUE=short
SWORKERS=2
SMEM=10
slurm_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "$(1)";

# SRC=french_trans-dev german_trans-dev italian_trans-dev
SRC=french_trans-dev

.SECONDARY:

# xml -> par only
# proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.par,$(shell find $(SRC) -name '*.xml')))
# xml -> par -> txt
proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.emb,$(shell find $(SRC) -name '*.xml')))
	touch $@

proc/%.par:	%.xml
	mkdir -p $(dir $@)
	$(call slurm_run,./book-stream.py '$<' > '$@')

proc/french_trans-dev/%.txt:	proc/french_trans-dev/%.par
	$(call slurm_run,./run_spacy.py '$<' > '$@')

proc/french_trans-dev/%_sents.txt:	proc/french_trans-dev/%.txt
	$(call slurm_run,./segment_sents.py '$<' > '$@')

proc/french_trans-dev/%.emb:	proc/french_trans-dev/%_sents.txt
	$(call slurm_run,./run_labse.py '$<' > '$@')

