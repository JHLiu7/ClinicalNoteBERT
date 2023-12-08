import torch, os, random
import torch.nn as nn 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


import pytorch_lightning as pl

from layers import init_model
from eval_utils import Evaluator



class ExpModule(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters(args)

        note_dim = _get_note_dim(self.hparams.note_encode_name)

        self.model = init_model(self.hparams, note_dim=note_dim)

        self.loss_module = nn.CrossEntropyLoss()
        self.scorer = Evaluator(self.hparams.task)

        self.batch_size = self.hparams.batch_size
        self.learning_rate = self.hparams.lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.wd)
        return optimizer

    def forward(self, x):

        if isinstance(x, tuple):
            logits = self.model(*x)
        else:
            logits = self.model(x)
        return logits 

    def step(self, batch):
        x, y, s = batch 

        logits = self.forward(x)
        loss = self.loss_module(logits, y)
        return loss, (logits, y)

    def training_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        self.log('train_loss', loss, on_epoch=True, batch_size=self.batch_size)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        self.log('valid_loss', loss, batch_size=self.batch_size)
        return {'loss': loss, 'logits':logits, 'y':y}

    def test_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        self.log('test_loss', loss, on_epoch=True, batch_size=self.batch_size)
        return {'loss': loss, 'logits':logits, 'y':y}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, s = batch
        logits = self.forward(x)
        return logits, y, s 

    def validation_epoch_end(self, list_of_dict):
        all_logits, all_y = [], []
        for d in list_of_dict:
            all_logits.append(d['logits'])
            all_y.append(d['y'])

        logits = torch.cat(all_logits).detach().cpu()
        y = torch.cat(all_y).cpu()
        
        scores, line = self.scorer.eval_all_scores(logits, y)
        score = scores[self.scorer.score_main]
        self.log('valid_score', score)

        if not self.hparams.silent:
            print(line)


def _get_note_dim(note_encode_name):

    if note_encode_name == 'avg_emb':
        encode_dim = 200
    elif ('lda' in note_encode_name) or ('doc2vec' in note_encode_name):
        encode_dim = int(note_encode_name.split('_')[1])
    elif 'tiny' in note_encode_name:
        encode_dim = 312
    else:
        encode_dim = 768

    return encode_dim
    

def init_trainer(args):

    RESULT_DIR = args.output_dir

    version = str(random.randint(0, 2000))
    log_dir = os.path.join(RESULT_DIR, 'log', args.modality)
    model_dir = os.path.join(RESULT_DIR, 'model', args.modality, args.task, version)

    device = None if args.device==-2 else [args.device] # device is int

    metric = 'valid_score'
    mode = 'max'

    logger = TensorBoardLogger(save_dir=log_dir, version=version, name=args.task)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, filename='{epoch}-{%s:.3f}' % metric, 
        monitor=metric, mode=mode, save_weights_only=True, 
    )
    early_stopping = EarlyStopping(
        monitor=metric, min_delta=0., patience=args.patience,
        verbose=False, mode=mode
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gpus=device,
        num_sanity_val_steps=0,
        max_epochs=args.epochs if not args.debug else 2, 
        enable_progress_bar=not args.silent,
    )
    return trainer

