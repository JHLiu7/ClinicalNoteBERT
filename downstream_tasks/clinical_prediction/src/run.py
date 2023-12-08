import logging
import pytorch_lightning as pl 

import numpy as np
from collections import defaultdict

from modules import ExpModule, init_trainer
from data_utils import ExpDataModule
from eval_utils import evaluate_predict_output

from options import args 


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')


def main_run(args, dm=None, seed=None):

    if seed is not None:
        pl.seed_everything(seed)
        logging.info(f'Set seed to {seed}')

    if dm is None:
        dm = ExpDataModule(args)
        dm.setup()

    model = ExpModule(args)

    if args.modality != 'text':
        model.model.GRUD._init_x_mean(dm.X_mean)

    trainer = init_trainer(args)
    trainer.fit(model, dm)

    output = trainer.predict(model, dataloaders=dm.test_dataloader(), ckpt_path='best')
    main_score, scores = evaluate_predict_output(output, args.task)

    return main_score, scores


def seed_runs(args):

    score_dicts = []
    seeds = [3407, 34071, 34072, 340, 1234]
    if args.debug: seeds = seeds[:2]
    for seed in seeds:
        main_score, scores = main_run(args, seed=seed)
        score_dicts.append(scores)

    collected = defaultdict(list)
    for d in score_dicts:
        for k,v in d.items():
            collected[k].append(v)
    
    metrics, nums = [], []
    for metric, scores in collected.items():
        metrics.append(metric.upper())
        nums.append( '{:.3f} ({:.4f})'.format(np.mean(scores), np.std(scores)) )

    ## print
    print(f'\nSpreaded scores for {args.task} with {args.modality} on {len(score_dicts)} runs')
    for k, v in collected.items():
        print(k)
        print([f'{i:.5f}' for i in v])
    print()

    title = f'\nAggregated scores on {len(score_dicts)} runs'
    l_metric, l_num = '\t'.join(metrics), '\t'.join(nums)

    print(title)
    print(l_metric)
    print(l_num)
    print()

    return '\n'.join([title, l_metric, l_num])


if __name__ == '__main__':
    main_run(args, seed=3407)
