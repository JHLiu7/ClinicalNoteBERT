
# ClinicalNoteBERT: More Performant and Efficient Encoders for Clinical Text

## 1. Overview

Using openly available clinical notes, we pretrain ClinicalNoteBERT, a series of encoders of three model sizes (110M, 67M, and 14.5M) that consider note contexts and variations during pretraining. We adopt a range of downstream applications to evaluate ClinicalNoteBERT, including tasks in fine-tuning, unsupervised semantic textual similarity, retrieval-augmented generation of LLMs, and unimodal and multimodal clinical predictions, and compare with strong baselines. Our models achieve better results than the baseline models of similar or larger sizes on various tasks and datasets. We find that different choices made during pretraining can lead to varied improvements for the downstream tasks. Our small and tiny versions of ClinicalNoteBERT maintain over 96% and 91% of the best performance with less than 61% and 14% of the parameters, respectively.


## 2. Pretrained Models

We provide five pretrained models with different pretraining recipes and in different sizes. 

|                            | # Params | Fine-tuning       | Download Links (🤗 HF) |
| -------------------------- | -------- | -------- | -------- |
| ClinicalNoteBERT-note-only | 110M     | 80.0     | [jhliu/ClinicalNoteBERT-base-note_only](https://huggingface.co/jhliu/ClinicalNoteBERT-base-note_only) |
| ClinicalNoteBERT-note-ntp  | 110M     | **80.6** | [jhliu/ClinicalNoteBERT-base-note_ntp](https://huggingface.co/jhliu/ClinicalNoteBERT-base-note_ntp) |
| ClinicalNoteBERT-base      | 110M     | 80.1     | [jhliu/ClinicalNoteBERT-base](https://huggingface.co/jhliu/ClinicalNoteBERT-base) |
| ClinicalNoteBERT-small     | 67M      | 78.1     | [jhliu/ClinicalNoteBERT-small](https://huggingface.co/jhliu/ClinicalNoteBERT-small) |
| ClinicalNoteBERT-tiny      | 14.5M    | 74.1     | [jhliu/ClinicalNoteBERT-tiny](https://huggingface.co/jhliu/ClinicalNoteBERT-tiny) |

The models are examined in other tasks to encode sequences of clinical text in different lengths. We applied unsupervised SimCSE training to better adapt the raw models to text encoders. Following the paper, we present the three variations in terms of training sequence lengths, releasing the following checkpoints evaluated on the STS, RAG, and Fusion (multimodal clinical prediction).

|                            | STS                                                          | RAG                                                          | Fusion                                                       |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ClinicalNoteBERT-note-only | 78.9 ([simcse_sentence](https://huggingface.co/jhliu/ClinicalNoteBERT-base-note_only-simcse_sentence)) | **14.0** ([simcse_segment](https://huggingface.co/jhliu/ClinicalNoteBERT-base-note_only-simcse_segment)) | 66.5 ([simcse_note](https://huggingface.co/jhliu/ClinicalNoteBERT-base-note_only-simcse_note)) |
| ClinicalNoteBERT-base      | **79.8** ([simcse_sentence](https://huggingface.co/jhliu/ClinicalNoteBERT-base-simcse_sentence)) | 12.3 ([simcse_segment](https://huggingface.co/jhliu/ClinicalNoteBERT-base-simcse_segment)) | 66.7 ([simcse_note](https://huggingface.co/jhliu/ClinicalNoteBERT-base-simcse_note)) |
| ClinicalNoteBERT-small     | 77.1 ([simcse_sentence](https://huggingface.co/jhliu/ClinicalNoteBERT-small-simcse_sentence)) | 11.4 ([simcse_segment](https://huggingface.co/jhliu/ClinicalNoteBERT-small-simcse_segment)) | **66.8** ([simcse_note](https://huggingface.co/jhliu/ClinicalNoteBERT-small-simcse_note)) |
| ClinicalNoteBERT-tiny      | 75.7 ([simcse_sentence](https://huggingface.co/jhliu/ClinicalNoteBERT-tiny-simcse_sentence)) | 8.9 ([simcse_segment](https://huggingface.co/jhliu/ClinicalNoteBERT-tiny-simcse_segment)) | 65.5 ([simcse_note](https://huggingface.co/jhliu/ClinicalNoteBERT-tiny-simcse_note)) |




Code for pretraining is also provided in `pretraining`, including the scripts we used to prepare the pretraining corpus and the ones we used for training. 

## 3. Downstream tasks

Code can be found in the directory `downstream_tasks` to curate datasets and run fine-tuning/training and evaluation. 


## Citation

Under review.

