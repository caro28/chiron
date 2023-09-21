QUEUE=short
SWORKERS=2
SMEM=10
slurm_run = srun --time 1-0 -p ${QUEUE} -N 1 -c $(SWORKERS) --mem=$(SMEM)G bash -c "$(1)";

# SRC=french_trans-dev german_trans-dev italian_trans-dev
SRC=french_trans-dev

.SECONDARY:

# xml -> par only
proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.par,$(shell find $(SRC) -name '*.xml')))
	touch $@

# xml -> par -> txt
# proc.done:	$(patsubst %,proc/%,$(patsubst %.xml,%.txt,$(shell find $(SRC) -name '*.xml')))

test.done:	$(patsubst %,test/%,$(patsubst %.par,%.txt,$(shell find $(./proc/french_trans-dev) -name '*.par')))
	touch $@

proc/%.par:	%.xml
	mkdir -p $(dir $@)
	$(call slurm_run,./book-stream.py '$<' > '$@')

# # # TODO: add lang rules here (with lang id)
test/%.txt:	%.par
	mkdir -p $(dir $@)
	$(call slurm_run,./run_spacy.py '$<' > '$@')

# proc/%.txt:	%.xml
# 	mkdir -p $(dir $@)
# 	proc/%.par:	%.xml
# 		$(call slurm_run,./book-stream.py '$<' > '$@')
# 	proc/%.txt:	proc/french_trans-dev/%.par
# 		$(call slurm_run,./run_spacy.py '$<' > '$@')
