### Mostly based on Huggingface run_mlm.py implementation
### See more in https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from packaging import version

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import datasets
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertForPreTrainingOutput
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizerFast,
    BertForMaskedLM,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import _torch_collate_batch, tolist

from utils import load_processed_data

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Should also contain tokenizer file."
        },
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Load model config and train from scratch."},
    )


@dataclass
class DataTrainingArguments:

    processed_file_dir: Optional[str] = field(
        default=None, metadata={"help": "The path for pre-loaded dataset that can be read with load_from_disk."}
    )
    train_seq_length: Optional[str] = field(
        default="256-512", metadata={"help": "Types of preprocessed notes used for pretraining, e.g., 256-512, 256-1024."}
    )

    validation_split: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The ratio of samples of the train set used as validation set in case there's no validation split"
        },
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    # Do not support processing files on the fly for now

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class VDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)

        batch['input_ids'] = inputs
        batch['labels'] = labels
        return batch
    

class ClinicalNoteBertPreTrainingHeadsNTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.note_type_discrimation = nn.Linear(config.hidden_size, 5)
        
    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        ntd_score = self.note_type_discrimation(pooled_output)
        return prediction_scores, (ntd_score)


class ClinicalNoteBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.config = config
        
        if version.parse(transformers.__version__) >= version.parse("4.13.0"):
            self.post_init()
        else:
            self.init_weights()

        self.cls = ClinicalNoteBertPreTrainingHeadsNTP(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            note_category=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):  

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, ntp_scores = self.cls(sequence_output, pooled_output)
            
            total_loss = None
            assert note_category is not None

            loss_fct = CrossEntropyLoss()

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            assert type(ntp_scores) is torch.Tensor

            ntp_loss = loss_fct(ntp_scores, note_category)

            total_loss = masked_lm_loss + ntp_loss

            if not return_dict:
                output = (prediction_scores, ntp_scores) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output
            
            return BertForPreTrainingOutput(
                loss=total_loss,
                prediction_logits=prediction_scores,
                seq_relationship_logits=ntp_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    ### -------------------
    ## Setup args & logging
    ### -------------------

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) <= 4 and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")



    ### -------------
    ## Detect ckpt
    ### -------------
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ### -------------
    ## Set seed
    ### -------------
    set_seed(training_args.seed)


    ### ---------------------
    ## Load tokenizer & model
    ### ---------------------
    tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config = BertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    if model_args.train_from_scratch:
        logger.info("Training new model from scratch")
        model = ClinicalNoteBertForPreTraining(config=config)
    else:
        model = ClinicalNoteBertForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir
        )

    ### ---------------
    ## Load datasets
    ### ---------------
    train_dataset, eval_dataset = load_processed_data(
        data_args, model_args, training_args, tokenizer, keep_note_label=True
    )

    logging.info("Datasets ready")

    data_collator = VDataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )



    ### -------------
    ## Train & Eval
    ### -------------
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
