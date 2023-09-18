QUEUE=short
SWORKERS=2
SMEM=10
slurm_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "$(1)";

SHELL=/bin/bash
CONDAROOT=/home/craig.car/miniconda3/
install: 
	use_lxml
	source $(CONDAROOT)/bin/activate && conda env create -p conda -f environment.yml && source deactivate;

# SRC=french_trans-dev german_trans-dev italian_trans-dev
SRC=french_trans-dev

.SECONDARY:

# proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.txt,$(shell find $(SRC) -name '*.xml')))
proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.par,$(shell find $(SRC) -name '*.xml')))
	touch $@

proc/%.par:	%.xml
	mkdir -p $(dir $@)
	$(call slurm_run,./book-stream.py '$<' > '$@')

# # TODO: fix paths
# # TODO: add lang rules here (with lang id)
# proc/%.txt:	proc/french_trans-dev/%.par
# 	mkdir -p $(dir $@)
# 	$(call slurm_run,./run_spacy.py '$<' > '$@')
