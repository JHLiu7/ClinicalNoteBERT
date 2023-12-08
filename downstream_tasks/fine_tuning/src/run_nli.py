""" Script for sent pair task, based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py."""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    DistilBertConfig,
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils import update_output_dir

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default='all', metadata={"help": "MedNLI"}
    )
    dataset_dir: str = field(
        default=None, metadata={"help": "Dir to raw datasets."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    save_model: bool = field(
        default=False, metadata={"help": "Save model if true."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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
    logger.info(f"Training/evaluation parameters {training_args}")


    # Set seed before initializing model.
    set_seed(training_args.seed)

    ### ---------
    ## Load data
    ### ---------

    data_path = os.path.join(data_args.dataset_dir, data_args.dataset_name)
    logging.info(f"Loaded raw dataset from {data_path}")
    
    assert data_args.dataset_name == 'MedNLI'

    raw_datasets = load_dataset('json', data_files={
        'train': os.path.join(data_path, 'mli_train_v1.jsonl'),
        'validation': os.path.join(data_path, 'mli_dev_v1.jsonl'),
        'test': os.path.join(data_path, 'mli_test_v1.jsonl')},
        cache_dir=os.path.join(data_args.dataset_dir, 'cache')
    )

    sentence1_key = 'sentence1'
    sentence2_key = 'sentence2'
    label_key = 'gold_label'

    label_to_id = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    num_labels = 3
    task_name = 'nli'
    is_regression = False

    col_to_remove = ['sentence1_binary_parse', 'sentence1_parse',
        'sentence2_binary_parse', 'sentence2_parse',
        'gold_label', 'pairID']





    ### ---------------------
    ## Load tokenizer & model
    ### ---------------------
    if 'distil' not in model_args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        config = DistilBertConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=model_args.cache_dir,
        )

    tokenizer_name_or_path = model_args.model_name_or_path
    if config.model_type in {"roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
        )

    if 'distil' not in model_args.model_name_or_path.lower():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = DistilBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    ### --------------
    ## Preprocess data
    ### --------------

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False


    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples[sentence1_key], examples[sentence2_key], padding=padding, max_length=max_seq_length, truncation=True)

        lbl = examples[label_key][0]
        if isinstance(lbl, str):
            result['label'] = [label_to_id[l] for l in examples[label_key]]
        elif isinstance(lbl, float):
            result['label'] = examples[label_key]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=col_to_remove+[sentence1_key, sentence2_key]
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 2):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if is_regression:
            metric = load_metric("pearsonr")
            scores = metric.compute(predictions=preds, references=p.label_ids)
            scores.update({"mse": ((preds - p.label_ids) ** 2).mean().item()})
            return scores
        else:

            # metric_f1 = load_metric('f1')
            # metric_acc= load_metric('accuracy')
            # result_f1 = metric_f1.compute(predictions=preds, references=p.label_ids)
            # result_acc= metric_acc.compute(predictions=preds, references=p.label_ids)
            # acc = (preds == p.label_ids).astype(np.float32).mean().item()

            f1_micro = f1_score(y_true=p.label_ids, y_pred=preds, average='micro')
            f1_macro = f1_score(y_true=p.label_ids, y_pred=preds, average='macro')

            acc = accuracy_score(y_true=p.label_ids, y_pred=preds)

            return {
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "accuracy": acc
            }
            # return {"acc": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    ### -----------------------
    ## Setup trainer & training
    ### -----------------------

    if data_args.max_seq_length > 128:
        task = f'NLI-{data_args.max_seq_length}'
    else:
        task = 'NLI'

    if training_args.load_best_model_at_end:
        training_args = update_output_dir(training_args, model_args, data_args, task, True)
    else:
        training_args = update_output_dir(training_args, model_args, data_args, task)




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        if model_args.save_model:
            trainer.save_model()
            trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    if training_args.do_predict:
        logger.info("*** Predict ***")

        # predict_dataset = predict_dataset.remove_columns("label")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.dataset_name}.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results {data_args.dataset_name} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = model.config.id2label[item]
                        writer.write(f"{index}\t{item}\n")





if __name__ == "__main__":
    main()
