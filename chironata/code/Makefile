QUEUE=short
SWORKERS=2
SMEM=10
slurm_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "$(1)";

SRC=french_trans-dev german_trans-dev italian_trans-dev

.SECONDARY:

proc.done:	$(patsubst %,data/%,$(patsubst %.xml,%.txt,$(shell find $(SRC) -name '*.xml')))
	touch $@

proc/%.par:	%.xml
	mkdir -p $(dir $@)
	$(call slurm_run,./book-stream.py '$<' > '$@')

# TODO: fix paths
# TODO: add lang rules here (with lang id)
data/french_trans-dev/%.txt:	proc/french_trans-dev/%.par
	mkdir -p $(dir $@)
	$(call slurm_run,./run_spacy.py '$<' > '$@')
