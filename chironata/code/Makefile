QUEUE=short
SWORKERS=2
SMEM=10
slurm_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "$(1)";

SRC=french_trans-dev german_trans-dev italian_trans-dev

.SECONDARY:

proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.par,$(shell find $(SRC) -name '*.xml')))
	touch $@

proc/%.par:	%.xml
	mkdir -p $(dir $@)
	$(call slurm_run,./book-stream.py '$<' > '$@')
