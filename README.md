# Chiron
Chiron is a tool for aligning pre-modern and literary texts with translations in multiple languages.

## Pipeline models
1. LaBSE, Feng et al. (2020)
* For embedding sentences
* Associated file: build_labse_embeds.py, using [Hugging Face implementation](https://huggingface.co/sentence-transformers/LaBSE)
* Input: text file to embed; output: LaBSE embeddings of 768 dimensions in binary file.
* LaBSE [paper](https://arxiv.org/abs/2007.01852): Feng, F., Yang, Y., Cer, D.M., Arivazhagan, N., & Wang, W. (2020). Language-agnostic BERT Sentence Embedding. *Annual Meeting of the Association for Computational Linguistics.*

2. Vecalign, Thompson (2019)
* For aligning two texts embedded at the sentence level
* Associated files: overlap.py, vecalign.py, score.py
* Vecalign GitHub: https://github.com/thompsonb/vecalign
* Vecalign [paper](https://aclanthology.org/D19-1136/): Thompson, B. (2019). Vecalign: Improved Sentence Alignment in Linear Time and Space. *Conference on Empirical Methods in Natural Language Processing.*

## Pipeline steps
1. Build overlaps files (from vecalign) for source text and translation
* File to run: overlap.py
* Input: source text or translation segmented at the sentence level
* Output: "concatenations of consecutive sentences" as explained on [Vecalign's GitHub](https://github.com/thompsonb/vecalign#embed-your-own-documents).

2. Build LaBSE embeddings of the overlaps files
* File to run: build_labse_embeds.py
* Input: overlaps text file
* Output: LaBSE embeddings in binary file of 768 dimensions, with 1 embedding per sentence concatenation in overlaps file

3. Align source text and translation (from vecalign)
* File to run: vecalign.py
* Input: LaBSE embeddings
* Output: sentence alignments written to stdout. For a detailed description of the results' format, see [Vecalign's GitHub](https://github.com/thompsonb/vecalign#run-vecalign-using-provided-embeddings).

## Evaluation
### Using sentence-level ground truth
* File to run: score_all.py
* Includes three scoring functions:
  * Vecalign's original strict scores (Precision, Recall, F1). Does not include Vecalign's original lax scores.
  * Chiron's new lax scores (Precision, Recall, F1)
  * Chiron's new strict score (Accuracy only)

### Chapter-level evaluation if sentence-level ground truth not available
* Example file: score_vec_rslts_chapter_level.ipynb
* Example based on aligning Thucydides' *The Peloponnesian War* against a [French translation](https://github.com/OpenGreekAndLatin/french_trans-dev/blob/master/thucydides_1863.xml)
* *Will upload data files*

## Align Texts Project
* Create an annotated dataset using the pipeline.
* Code and data saved in [align_texts_project](https://github.com/caro28/chiron/tree/main/align_texts_project)

## Related experiments
*Coming soon: folder with code, data, results of "background experiments" run on Crito and Thucydides, using LASER and LaBSE.*

## Installation
1. To use Vecalign, see list of dependencies on [Vecalign's GitHub](https://github.com/thompsonb/vecalign#build-vecalign)
2. To use LaBSE, see instructions on [Hugging Face](https://huggingface.co/sentence-transformers/LaBSE#usage-sentence-transformers)
