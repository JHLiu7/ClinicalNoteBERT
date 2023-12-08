import datasets, os
from datasets import load_dataset, concatenate_datasets


def load_processed_data(data_args, model_args, training_args, tokenizer, keep_note_label=False, seed=42):

    # processed_datasets = datasets.load_from_disk(dataset_path=data_args.processed_file_dir)

    # load preprocessed notes in different categories
    NOTE_CATEGORIES = ['discharge', 'radiology', 'nursing', 'physician', 'others']

    all_datasets = []
    for category in NOTE_CATEGORIES:
        dset = datasets.load_from_disk(dataset_path=os.path.join(
            data_args.processed_file_dir, f"{category}-{data_args.train_seq_length}"
        ))

        all_datasets.append(dset)

    processed_datasets = concatenate_datasets(all_datasets).shuffle(seed=seed)

    if keep_note_label == False and 'note_category' in processed_datasets.column_names:
        processed_datasets = processed_datasets.remove_columns(['note_category'])


    # load/create validation set
    if hasattr(data_args, "validation_file") and data_args.validation_file:
        # use validation file if provided
        # process on the fly
        raw_datasets = load_dataset('text', data_files={'validation': data_args.validation_file}, cache_dir=model_args.cache_dir)
        text_column_name = "text"
        padding = "max_length"
        max_seq_length = data_args.max_eval_length
        # line by line
        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )

        eval_dataset = tokenized_datasets["validation"]
        train_dataset = processed_datasets

    else:
        split_datasets = processed_datasets.train_test_split(data_args.validation_split)

        eval_dataset = split_datasets['test']
        train_dataset = split_datasets['train']

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    
    return train_dataset, eval_dataset
