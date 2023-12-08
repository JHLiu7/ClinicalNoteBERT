import subprocess, os, argparse

from utils import _get_abbrv_from_model_name_or_path
from utils_re import evaluate_labeler


def _create_training_config(model_path, config_dir, epoch=30, jsonnet_file='radgraph.jsonnet'):

    new_lines = []
    for line in open(os.path.join(config_dir, jsonnet_file), 'r').readlines():
        if line.startswith('    bert_model'):
            line = f'    bert_model: "../{model_path}",'
        if line.startswith('    num_epochs'):
            line = f'    num_epochs: {epoch},'

        new_lines.append(line)

    model_name = _get_abbrv_from_model_name_or_path(model_path)

    jsonnet_path = os.path.join(config_dir, f'{model_name}.jsonnet')

    with open(jsonnet_path, 'w') as out:
        out.write(''.join(new_lines))

    return jsonnet_path, model_name

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/FT/", help="directory to radgraph dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="model name or path")

    parser.add_argument("--output_dir", type=str, default="results", help="directory to radgraph dataset")
    args = parser.parse_args()
    
    
    # model_path = "pretrained_models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    model_path = args.model_name_or_path
    dataset_dir = os.path.abspath(os.path.join(args.dataset_dir, 'radgraph'))
    EPOCH = 30


    # create jsonnet config in the right path
    config_dir = 'dygiepp/training_config'
    jsonnet_path, model_name = _create_training_config(model_path, config_dir, epoch=EPOCH)
    jsonnet_path = jsonnet_path.strip('dygiepp/')

    TMP_CKPT_DIR = 'cache/dygiepp_ckpt'

    model_dir =  os.path.abspath(os.path.join(TMP_CKPT_DIR, model_name))
    output_dir = os.path.abspath(os.path.join(args.output_dir, model_name, 'RE'))
    os.makedirs(output_dir, exist_ok=True)

    # train model
    bashCommand = f"allennlp train {jsonnet_path} --serialization-dir {model_dir} --include-package dygie"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, cwd="dygiepp")
    output, error = process.communicate()


    # predict 
    ckpt_path = os.path.join(model_dir, 'model.tar.gz')

    test_files = ['test.jsonl', 'test-MIMIC-CXR-labeler_1.jsonl', 'test-MIMIC-CXR-labeler_2.jsonl',
                  'test-CheXpert-labeler_1.jsonl', 'test-CheXpert-labeler_2.jsonl']
    pred_files = ['pred.jsonl', 'pred-MIMIC-CXR-labeler_1.jsonl', 'pred-MIMIC-CXR-labeler_2.jsonl',
                 'pred-CheXpert-labeler_1.jsonl', 'pred-CheXpert-labeler_2.jsonl']
    

    test_files = [os.path.join(dataset_dir, f) for f in test_files]
    pred_files = [os.path.join(output_dir, f) for f in pred_files]
    
    for test_file, pred_file in zip(test_files, pred_files):
        
        predCommand = \
        f"allennlp predict {ckpt_path} {test_file} --predictor dygie --include-package dygie --use-dataset-reader --output-file {pred_file} --cuda-device 0 --silent"

        process = subprocess.Popen(predCommand.split(), stdout=subprocess.PIPE, cwd="dygiepp")
        output, error = process.communicate()



    # evaluate
    mimic_micro, mimic_macro = evaluate_labeler(
        json_file1=os.path.join(output_dir, 'pred-MIMIC-CXR-labeler_1.jsonl'),
        json_file2=os.path.join(output_dir, 'pred-MIMIC-CXR-labeler_2.jsonl')
    )

    chxpert_micro, chxpert_macro = evaluate_labeler(
        json_file1=os.path.join(output_dir, 'pred-CheXpert-labeler_1.jsonl'),
        json_file2=os.path.join(output_dir, 'pred-CheXpert-labeler_2.jsonl')
    )

    stdout = f"""
Evaluate results:

Two scores
MIMIC-CXR-Micro\tCheXpert-Micro
{mimic_micro:.3f}\t{chxpert_micro:.3f}


Four scores
MIMIC-CXR-Micro\tMIMIC-CXR-Macro\tCheXpert-Micro\tCheXpert-Macro
{mimic_micro:.3f}\t{mimic_macro:.3f}\t{chxpert_micro:.3f}\t{chxpert_macro:.3f}

    """

    with open(os.path.join(output_dir, 'eval_res.txt'), 'w') as out:
        out.write(stdout)

