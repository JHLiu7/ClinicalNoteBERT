import argparse


def add_task_args(parent_parser):
    parser = parent_parser.add_argument_group("Task")

    parser.add_argument("--task", type=str, required=True, choices=['mort_hosp', 'los_3', 'drg'])

    parser.add_argument("--modality", type=str, default="text", choices=['text', 'struct', 'both'])

    parser.add_argument("--note_encode_dir", type=str, default="data_notes_encoded")
    parser.add_argument("--note_encode_name", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--root_dir", type=str, default="/scratch/punim1362/jinghuil1/TmpCBERT/data/EB/clinical_pred")
    parser.add_argument("--output_dir", type=str, default="/scratch/punim1362/jinghuil1/TmpCBERT/cache/clinical_pred/output")

    return parent_parser


def add_hp_args(parent_parser):
    parser = parent_parser.add_argument_group("Hyperparameter")

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_size_txt", type=int, default=128)
    parser.add_argument("--hidden_size_ts", type=int, default=128)

    return parent_parser


def add_train_args(parent_parser):
    parser = parent_parser.add_argument_group("Train")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--silent', action="store_const", const=True, default=False)
    parser.add_argument('--debug', '-D', action="store_const", const=True, default=False)

    return parent_parser


def add_tune_args(parent_parser):
    parser = parent_parser.add_argument_group("Tune")
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--gpus_per_trial', default=0.2, type=float)
    return parent_parser


parser = argparse.ArgumentParser()

parser = add_task_args(parser)
parser = add_hp_args(parser)
parser = add_train_args(parser)
parser = add_tune_args(parser)

args = parser.parse_args()
