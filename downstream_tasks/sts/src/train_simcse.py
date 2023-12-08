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
from datasets import load_dataset
import torch.distributed as dist

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import random

import transformers
from transformers.models.bert.modeling_bert import SequenceClassifierOutput, BertPreTrainedModel, BertForPreTrainingOutput
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
    RobertaModel,
    RobertaPreTrainedModel,
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import _torch_collate_batch, tolist

logger = logging.getLogger(__name__)

from simcse_utils import get_tokenized_lines

class SimCSE_BERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.config = config

        if version.parse(transformers.__version__) >= version.parse("4.13.0"):
            self.post_init()
        else:
            self.init_weights()

        self.temp  = None # init later

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):



        # get embeddings 
        outputs1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True
        )
        outputs2 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True
        )

        # get cls pooled rep 
        pooler_output1 = outputs1.last_hidden_state[:, 0]
        pooler_output2 = outputs2.last_hidden_state[:, 0]

        z1 = self.mlp(pooler_output1)
        z2 = self.mlp(pooler_output2)

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)


        cos = nn.CosineSimilarity(dim=-1)
        cos_sim = cos(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp

        loss_func = nn.CrossEntropyLoss()
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)

        loss = loss_func(cos_sim, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=None,
            attentions=None
        )



class SimCSE_Roberta(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.config = config

        if version.parse(transformers.__version__) >= version.parse("4.13.0"):
            self.post_init()
        else:
            self.init_weights()

        self.temp  = None # init later

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        # ori_input_ids = input_ids
        # batch_size = input_ids.size(0)
        # num_doc = input_ids.size(1)

        # get embeddings 
        outputs1 = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True
        )
        outputs2 = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True
        )

        # get cls pooled rep 
        pooler_output1 = outputs1.last_hidden_state[:, 0]
        pooler_output2 = outputs2.last_hidden_state[:, 0]

        z1 = self.mlp(pooler_output1)
        z2 = self.mlp(pooler_output2)

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)


        cos = nn.CosineSimilarity(dim=-1)
        cos_sim = cos(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp

        loss_func = nn.CrossEntropyLoss()
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)

        loss = loss_func(cos_sim, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=None,
            attentions=None
        )




@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Should also contain tokenizer file."
        },
    )

    temperature: float = field(
        default=0.05, metadata={"help": "Temp for InfoNCE"}
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class DataTrainingArguments:

    raw_text_dir: Optional[str] = field(
        default=None, metadata={"help": "The path for raw text lines."}
    )

    sequence_type: str = field(
        default=None,
        metadata={
            "help": "Seq type: sent, seg, note."
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def main():

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
    if 'roberta' in model_args.model_name_or_path.lower():
        logger.info('LLLah!!')
        tokenizer = RobertaTokenizerFast.from_pretrained(model_args.model_name_or_path, 
            local_files_only=True,
            cache_dir=model_args.cache_dir
        )
        model = SimCSE_Roberta.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            local_files_only=True,
        )
    else:
        tokenizer = BertTokenizerFast.from_pretrained(model_args.model_name_or_path, 
            local_files_only=True,
            cache_dir=model_args.cache_dir
        )
        model = SimCSE_BERT.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            local_files_only=True,
        )
    model.temp = model_args.temperature 


    ### ---------------
    ## Load datasets
    ### ---------------


    batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps = training_args.max_steps
    samples = batch_size * steps

    train_dataset = get_tokenized_lines(
        tokenizer=tokenizer,
        RAW_TEXT_DIR=data_args.raw_text_dir,
        NUM_SEQS=samples,
        LINE_TYPE=data_args.sequence_type,
        cache_dir=model_args.cache_dir,
        seed=training_args.seed
    )

    logging.info("Datasets ready")
    logging.info(f"{len(train_dataset):,} lines (expecting {samples:,})")



    ### ---------------
    ## Setup trainer
    ### ---------------

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )


    ### -------------
    ## Train 
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


if __name__ == "__main__":
    main()

