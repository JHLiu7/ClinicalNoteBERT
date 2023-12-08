import os, sys, math, copy, logging, warnings, shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from modules import ExpModule, init_trainer
from data_utils import ExpDataModule
from eval_utils import evaluate_predict_output

from run import seed_runs

from options import args 

warnings.filterwarnings("ignore")
os.environ["SLURM_JOB_NAME"] = "bash"

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')



def train_with_tune(config, args, dm, num_gpus):

    args = copy.deepcopy(args)
    args.silent = True
    vars(args).update(config)

    model = ExpModule(args)
    if args.modality != 'text':
        model.model.GRUD._init_x_mean(dm.X_mean)

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[EarlyStopping(monitor='valid_score', min_delta=0.00, 
                    patience=args.patience, verbose=False, mode="max"),
            TuneReportCallback({"val_score": 'valid_score'}, on='validation_end')],
        max_epochs=200 if not args.debug else 2,
        gpus=math.ceil(num_gpus),
        enable_progress_bar=False,
    )


    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    trainer.fit(model, train_dataloader, val_dataloader)



def tune_configs(configs, args):
    logging.info('Start')

    root_dir = os.path.abspath(args.root_dir)

    dm = ExpDataModule(args)
    dm.setup()

    select_mode = 'max'
    asha_scheduler = ASHAScheduler(max_t=50, grace_period=5, reduction_factor=2)
    trial_name = f'tune_with_asha_{args.modality}/{args.task}'


    analysis = tune.run(
        tune.with_parameters(train_with_tune, args=args, dm=dm, num_gpus=args.gpus_per_trial),
        resources_per_trial={'cpu':1, 'gpu':args.gpus_per_trial},
        metric='val_score', mode=select_mode, config=configs,
        max_concurrent_trials=8,
        num_samples=args.num_samples, name=trial_name, scheduler=asha_scheduler, local_dir=root_dir
    )

    best_config = analysis.best_config
    best_score = analysis.best_result['val_score']

    logging.info("Best hyperparameters found were: " + str(best_config))
    logging.info("Best val score: {:.4f}".format(best_score))
    
    return best_config


def get_configs(args):
    
    configs = {
        "dropout": tune.grid_search([0.1, 0.2, 0.4, 0.6]),
        "wd": tune.grid_search([0, 1e-4, 1e-3]),
    }

    if args.modality == 'struct':
        configs.update({
            "hidden_size_ts": tune.grid_search([64, 128, 256]),
        })

    elif args.modality == 'text':
        configs.update({
            "hidden_size_txt": tune.grid_search([64, 128, 256]),
        })

    elif args.modality == 'both':
        configs.update({
            "hidden_size_txt": tune.grid_search([64, 128, 256]),
        })
    
    else:
        raise NotImplementedError

    if args.debug: configs = {
        "dropout": tune.grid_search([0.1, 0.2]),
    }

    return configs 



def main():
    configs = get_configs(args)
    # tune
    best_config = tune_configs(configs, args)

    # run
    logging.info('Train with best config from scratch')
    vars(args).update(best_config)

    out = seed_runs(args)

    print(f"\nFor MODALITY - {args.modality}; TASK - {args.task}; NAME - {args.note_encode_name}; with config {best_config}")


    ROOT_DIR = '/data/scratch/projects/punim1362/jinghuil1/TmpCBERT/'
    if args.model_name is not None:
        encode_type = args.note_encode_name.split('-')[-1]
        model_name = args.model_name
    else:
        model_name, encode_type = args.note_encode_name.split('-')
    folder = os.path.join(ROOT_DIR, 'results', model_name, 'ClinicalPred', encode_type)
    os.makedirs(folder, exist_ok=True)
    fpath = os.path.join(folder, f'{args.modality}-{args.task}.txt')

    with open(fpath, 'w') as f:
        f.write(args.note_encode_name)
        f.write('\n')
        f.write(out)

    logging.info(f'Write results to {fpath}')

if __name__ == '__main__':
    main()