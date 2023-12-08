import os 

def _get_abbrv_from_model_name_or_path(model_path):

    if os.path.isdir(model_path):
        model_name = model_path.strip('/').split('/')[-1]


    if model_name == 'bert-base-cased':
        name = 'bert'
    elif model_name == '':
        name = 'ClinicalBERT'
    elif model_name == 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':
        name = 'PubMedBERT'

    else:
        name = model_name

    return name 


def update_output_dir(training_args, model_args, data_args, task='NER', early_stopping=False):
    
    bsize = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    epoch = training_args.num_train_epochs

    lr = training_args.learning_rate
    schedule = training_args.lr_scheduler_type.value
    
    run_name = f'{int(epoch)}epoch-{bsize}batch-{schedule}{lr}'

    model_name = _get_abbrv_from_model_name_or_path(model_args.model_name_or_path)

    task_name = f"{task}_{data_args.dataset_name}"

    if early_stopping:
        training_args.output_dir = os.path.join(
            training_args.output_dir, model_name, task_name+'_early'
        )
    else:
        training_args.output_dir = os.path.join(
            training_args.output_dir, model_name, task_name, run_name
        )

    return training_args


