import datasets
import torch, os, sys, argparse, logging

from datasets import load_dataset
from tqdm import tqdm

import numpy as np 
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSeq2SeqLM
)
from utils import get_index_name

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--retriever_model_name_or_path", type=str, default="")

    parser.add_argument("--datastore_path", type=str, default="")

    parser.add_argument("--embeddings_dir", type=str, default="data/RA/embeddings")
    parser.add_argument("--faiss_dir", type=str, default="data/RA/faiss")

    parser.add_argument("--num_proc", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--retriever_ix", type=int, default=0)

    parser.add_argument("--debug", "-D", action='store_true', default=False)

    args = parser.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)
    os.makedirs(args.faiss_dir, exist_ok=True)

    # quick fix on model name

    models = [
        "RoBERTa-base-PM-M3-Voc-train-longer",
        "PubMedBERT",
        "base_basic-25k",
        "srun10k_base_s",
        "srun10k_base_ss",
        "srun10k_base_ssn",
    ]
    if args.retriever_model_name_or_path != '':
        model_name_or_path = args.retriever_model_name_or_path
    else:
        assert args.retriever_ix > 0
        seq_type = 'segment' if 'chunks_seg' in  args.datastore_path else 'sentence'
        model_name_or_path = os.path.join(
            'SimCSE', models[args.retriever_ix-1], f'{seq_type}-best'
        )


    # load dataset
    raw_dataset = datasets.load_from_disk(args.datastore_path)

    if args.debug: raw_dataset = raw_dataset.select(range(10_000))


    # check faiss
    index_path = os.path.join(
        args.faiss_dir, get_index_name(model_name_or_path, args.datastore_path, len(raw_dataset))
    )
    if os.path.isfile(index_path):
        print(f'stopped; {index_path} existed')
        sys.exit()


    # load retriever
    cache_dir = "tmp"
    torch.set_grad_enabled(False)
    retriever_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    retriever_encoder   = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir).eval().cuda()


    # tokenize
    def tokenize_function(examples):
        examples = retriever_tokenizer(examples["text"], 
            padding="max_length", truncation=True, max_length=512)
        return examples

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=["text"],
        num_proc=args.num_proc
    )
    tokenized_dataset.set_format(type="torch", device="cuda")


    # encode
    batch_size = args.batch_size
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size) 
    num_batch = int(len(tokenized_dataset) / batch_size)

    embeddings = []

    for batch in tqdm(data_loader, total=num_batch):
        out = retriever_encoder(**batch)
        emb = out.last_hidden_state[:, 0].cpu()

        embeddings.append(emb)

    embeddings = torch.cat(embeddings, 0).numpy()

    emb_path = os.path.join(
        args.embeddings_dir, get_index_name(model_name_or_path, args.datastore_path, len(raw_dataset), True)
    )

    np.save(emb_path, embeddings)

    logging.info(f'Saved embeddings to {emb_path}')


if __name__ == "__main__":
    main()