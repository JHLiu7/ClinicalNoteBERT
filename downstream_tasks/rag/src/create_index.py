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


from create_embeddings import get_index_name




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--retriever_model_name_or_path", type=str, default="")

    parser.add_argument("--datastore_path", type=str, default="")

    parser.add_argument("--embeddings_dir", type=str, default="data/RA/embeddings")
    parser.add_argument("--faiss_dir", type=str, default="data/RA/faiss")


    parser.add_argument("--debug", "-D", action='store_true', default=False)

    args = parser.parse_args()


    # load dataset
    raw_dataset = datasets.load_from_disk(args.datastore_path)

    if args.debug: raw_dataset = raw_dataset.select(range(10_000))


    # check faiss
    model_name_or_path = args.retriever_model_name_or_path

    index_path = os.path.join(
        args.faiss_dir, get_index_name(model_name_or_path, args.datastore_path, len(raw_dataset))
    )
    if os.path.isfile(index_path):
        print(f'stopped; {index_path} existed')
        sys.exit()


    emb_path = os.path.join(
        args.embeddings_dir, get_index_name(model_name_or_path, args.datastore_path, len(raw_dataset), True)
    )
    embeddings = np.load(emb_path)
    embeddings = [i for i in embeddings]


    # add faiss
    logging.info(f'create faiss index')

    ds_with_embeddings = raw_dataset.add_column('embeddings', embeddings)
    ds_with_embeddings.add_faiss_index(column='embeddings', device=0)

    ds_with_embeddings.save_faiss_index('embeddings', index_path)

    logging.info(f'saved faiss index to {index_path}')


if __name__ == "__main__":
    main()