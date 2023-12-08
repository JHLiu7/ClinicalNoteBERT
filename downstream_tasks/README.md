## Downstream Tasks

### 1. Fine-tuning on standard clinical NLP datasets

We use `MedNLI`, `i2b2-2010`, `i2b2-2012`, `RadGraph`, `emrQA`, and `RadQA` datasets for fine-tuning. These datasets need to be found in their corresponding webpages. For `i2b2` datasets, we follow the [ClinicalBERT repo](https://github.com/EmilyAlsentzer/clinicalBERT) to preprocess the data. For `emrQA`, we follow [CliniRC repo](https://github.com/xiangyue9607/CliniRC) to downsample the `medication` subset for our evaluaiton. Other datasets should be readily usable after downloading them from the original sources. 

After preparing the data, one can follow the `fine_tuning/finetune_tasks.sh` to fine-tune the model for the NLI, NER, and QA tasks. For RE, we rely on [DyGIE++](https://github.com/dwadden/dygiepp) to perform end-to-end relation extraction. This requires setting up a new environment with dependencies listed in `fine_tuning/re_requirements.txt`, after which one can run `fine_tuning/finetune_re.sh`. 



### 2. Encoding sentences for unsupervised STS

We evaluate ClinicalNoteBERT as a text sequence encoder with several downstream tasks. The first one is semantic textual similarity (STS), which relies on the [ClinicalSTS](https://pubmed.ncbi.nlm.nih.gov/33245291/) dataset. We apply unsupervised [SimCSE](https://github.com/princeton-nlp/SimCSE) training to adapt the raw BERT models to better encoders, but we varied the text sequence length during this training process. One can evaluate our released checkpoint using `sts/evaluate_sts.sh` or use our code to adapt any other model with SimCSE.



### 3. Encoding segments/chunks for RAG

We apply ClinicalNoteBERT as a retriever to perform in-context augmentation for GPT2 and Llama2, evaluating their generation by perplexity. We sample test notes from MIMIC-III and MIMIC-IV and use non-overlapping notes (from non-overlapping patients for MIMIC-III and non-overlapping period for MIMIC-IV; more detail in the paper) as datastore. Run `rag/data/run.sh` to prepare the data, and run `rag/evaluate_rag.sh` for evaluation, which creates index and calculate perplexity over two testsets with four LLMs (GPT2-base, GPT2-xl, Llama2-7b, Llama2-13b).



### 4. Encoding notes for clinical predictions

We focus on three clinical prediction tasks and use ClinicalNoteBERT to encode clinical notes, which are used as input for the predictive analysis. These three tasks are based on two cohorts, which are provided in`clinical_prediction/data/cohorts`. Then clinical notes need to be extracted for these cohorts using `clinical_prediction/data/extract_notes.py`.

For modeling, use `clinical_prediction/clinical_pred_with_note_embeddings.sh` to encode notes, tune hyperparamters, and train the models on the three prediction tasks: in-hopsital mortality, length-of-stay over 3 days, and discharge diagnosis-related group (DRG). Notice the hyperparameter tuning process relies on [ray-tune](https://github.com/ray-project/ray/tree/master), and one may want to set up a new environment based on `cp_requirements.txt`.  



### 5. Encoding notes for multimodal fusion

We combine the note embeddings with structured clinical values to jointly model the three clinical prediction tasks. We follow [MIMIC-Extract](https://github.com/MLforHealth/MIMIC_Extract) to extract and curate 104 the clinical measurements, which are modeled as time-series. As the cohort to predict patient mortality and length-of-stay is sourced based on MIMIC-Extract, one can directly extract features from their released benchmark file using `clinical_prediction/data/prepare_from_original.py`. For the DRG cohort, we adapt the original script to extract from raw MIMIC database. This is achieved by running `clinical_prediction/data/extract_hourly_fts.py` and `clinical_prediction/data/prepare_hourly_fts.py`. With both notes and measurements ready, run `clinical_prediction/clinical_pred_with_fusion.sh`. 
