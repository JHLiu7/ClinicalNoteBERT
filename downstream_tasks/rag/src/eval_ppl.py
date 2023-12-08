import datasets
import torch, os, argparse, logging, copy

from datasets import load_dataset
from tqdm import tqdm

import numpy as np 
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSeq2SeqLM,
    LlamaTokenizerFast
)
from utils import get_index_name, get_outfile_path


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--lm_model_name_or_path", type=str, required=True)
    parser.add_argument("--retriever_model_name_or_path", type=str, required=True)

    parser.add_argument("--datastore_path", type=str, default="")
    parser.add_argument("--eval_data_path", type=str, default="")

    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--topK", type=int, default=5)

    parser.add_argument("--write_results", "-R", action="store_const", const=True, default=False)

    args = parser.parse_args()

    ppl_base, ppl_rag, ppl_rag_plus = evaluate_ppl(
        args.lm_model_name_or_path,
        args.retriever_model_name_or_path,
        args.datastore_path,
        args.eval_data_path,
        args,
        cache_dir='tmp',
        llama_cache_dir='/data/scratch/projects/punim1362/jinghuil1/Llama/'
    )

    if args.write_results:

        outfile = get_outfile_path(args)
        
        with open(outfile, 'w') as f:
            f.write(f"{ppl_base} {ppl_rag} {ppl_rag_plus}")



def evaluate_ppl(
    lm_model_name_or_path,
    retriever_model_name_or_path,
    datastore_path,
    eval_data_path,
    config,
    cache_dir,
    llama_cache_dir,
): 
    cache_dir = 'tmp'
    torch.set_grad_enabled(False)

    # get base model for lm
    if 'llama' in lm_model_name_or_path:
        model_name = lm_model_name_or_path
        cache_folder = os.path.join(llama_cache_dir, f'{model_name}')

        tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=cache_folder)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                device_map='cuda',
                                                torch_dtype=torch.float16, 
                                                cache_dir=cache_folder)
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_model_name_or_path, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(lm_model_name_or_path, cache_dir=cache_dir).eval().cuda()

    logging.info(f'Base model loaded: {lm_model_name_or_path}')

    # get test data
    eval_text_string = prepare_eval_dataset(eval_data_path)
    encodings = tokenizer(eval_text_string, add_special_tokens=False, return_tensors="pt")

    logging.info(f'Test data from {eval_data_path} loaded and encoded')


    # retriever and datastore
    retriever_tokenizer, retriever_encoder, ds_with_embeddings = prepare_index_and_retriever(retriever_model_name_or_path, cache_dir, datastore_path)

    logging.info(f'Retriever ({retriever_model_name_or_path}) and datastore ({len(ds_with_embeddings)} entries) ready')


    # eval ppl w/o retrieval
    ppl_base = eval_dataset_with_retrieval(
        eval_model=model,
        eval_tokenizer=tokenizer,
        encodings=encodings,
        retriever=None,
        retriever_tokenizer=None,
        faiss_dataset=None,
        stride=config.stride, max_length=config.max_length
    )
    logging.info(f'ppl_base: {ppl_base}')

    # eval ppl w/ retrieval
    ppl_rag = eval_dataset_with_retrieval(
        eval_model=model,
        eval_tokenizer=tokenizer,
        encodings=encodings,
        retriever=retriever_encoder,
        retriever_tokenizer=retriever_tokenizer,
        faiss_dataset=ds_with_embeddings,
        stride=config.stride, max_length=config.max_length
    )
    logging.info(f'ppl_rag: {ppl_rag}')

    # eval ppl w/ retrieval ensemble
    assert config.topK > 1
    ppl_rag_plus = eval_dataset_with_retrieval(
        eval_model=model,
        eval_tokenizer=tokenizer,
        # eval_text_string=eval_text_string,
        encodings=encodings,
        retriever=retriever_encoder,
        retriever_tokenizer=retriever_tokenizer,
        faiss_dataset=ds_with_embeddings,
        stride=config.stride, max_length=config.max_length, topK=config.topK
    )
    logging.info(f'ppl_rag top{config.topK}: {ppl_rag_plus}')

    ppl_base, ppl_rag, ppl_rag_plus = [t.item() for t in [ppl_base, ppl_rag, ppl_rag_plus]]

    return ppl_base, ppl_rag, ppl_rag_plus


def prepare_index_and_retriever(model_name_or_path, cache_dir, datastore_path, faiss_dir='data/RA/faiss'):
    retriever_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    retriever_encoder   = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir).eval().cuda()


    ds_with_embeddings = datasets.load_from_disk(datastore_path)

    index_path = os.path.join(
        faiss_dir, get_index_name(model_name_or_path, datastore_path, len(ds_with_embeddings))
    )
    ds_with_embeddings.load_faiss_index('embeddings', index_path)

    return retriever_tokenizer, retriever_encoder, ds_with_embeddings


def prepare_eval_dataset(eval_data_path):
    # eval_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='validation[:100]', cache_dir="tmp")

    eval_dataset = datasets.load_from_disk(eval_data_path)

    eval_text_string = "".join([x["text"] if x["text"] else " \n" for x in eval_dataset])
    return eval_text_string



def _encode_query(model, tokenizer, query):
    device = model.device

    # output = model(**inp.to(device))

    enc = tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    position_ids = torch.arange(enc.input_ids.size(1)).unsqueeze(0).type_as(enc.input_ids)
    enc['position_ids'] = position_ids

    output = model(**enc.to(device))

    if hasattr(output, 'last_hidden_state'):
        rep = output.last_hidden_state[:, 0].squeeze()
    else:
        rep = output[0][0]
    return rep


def eval_dataset_with_retrieval(
    eval_model, 
    eval_tokenizer,
    encodings,
    retriever,
    retriever_tokenizer,
    faiss_dataset,
    topK=1,
    emb_col='embeddings',
    text_col='text',
    stride=64,
    max_length=128,
    device='cuda',
):

    seq_len = encodings.input_ids.size(1)

    if retriever is not None and retriever_tokenizer is not None and faiss_dataset is not None:
        USE_RETRIEVAL=True
    else:
        USE_RETRIEVAL=False

    def _one_pass(input_ids, trg_len):

        input_ids = input_ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = eval_model(input_ids, labels=target_ids)

            if trg_len < max_length:
                neg_log_likelihood = outputs.loss * trg_len
                lm_logits = outputs.logits[..., -trg_len-1:-1, :]
                labels = target_ids[..., -trg_len:]
            else:
                neg_log_likelihood = outputs.loss * (max_length - 1)
                lm_logits = outputs.logits[..., -max_length:-1, :]
                labels = target_ids[..., -max_length+1:]
            
            neg_log_likelihood = neg_log_likelihood.squeeze()
            # neg_log_likelihood = outputs.loss

        return neg_log_likelihood, lm_logits, labels

    def aug_query_with_ctx(query, ctx):


        ctx_ids = eval_tokenizer(ctx, add_special_tokens=False, return_tensors="pt").input_ids
        new_input_ids = torch.concat([ctx_ids, input_ids], 1)
        return new_input_ids[:, -eval_model.config.max_position_embeddings:]

        

    def _get_loss(lm_logits, labels):
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1)
        ).cpu()
        return loss
        
    logging.info('start eval')

    nlls = []
    all_token_ppls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]


        if USE_RETRIEVAL:
            # retrieve 
            query = eval_tokenizer.decode(input_ids.squeeze())
            query_emb = _encode_query(retriever, retriever_tokenizer, query)
            query_emb = query_emb.cpu().numpy()
            _, retrieved_samples = faiss_dataset.get_nearest_examples(emb_col, query_emb, k=topK)
            if topK == 1:
                ctx = retrieved_samples[text_col][0]

                new_input_ids = aug_query_with_ctx(query, ctx)

                neg_log_likelihood, lm_logits, labels = _one_pass(new_input_ids, trg_len)
                loss = _get_loss(lm_logits, labels)

                nlls.append(neg_log_likelihood)
                all_token_ppls.append(loss)
            
            else:
                # aggregate logits over multiple retrieved chunks
                lm_logits_set = []
                labels_set = []
                for ctx in retrieved_samples[text_col]:
                    
                    new_input_ids = aug_query_with_ctx(query, ctx)


                    _, lm_logits, labels = _one_pass(new_input_ids, trg_len)

                    lm_logits_set.append(lm_logits)
                    labels_set.append(labels)

                lm_logits_agg = torch.stack(lm_logits_set).mean(0)
                labels = labels_set[0]
                # assert 
                loss = _get_loss(lm_logits_agg, labels)
                all_token_ppls.append(loss)

        else:
            # no retrieval
            neg_log_likelihood, lm_logits, labels = _one_pass(input_ids, trg_len)
            loss = _get_loss(lm_logits, labels)

            nlls.append(neg_log_likelihood)
            all_token_ppls.append(loss)


        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl_over_logits = np.exp(sum([sum(x) for x in all_token_ppls]) / seq_len)


    if not USE_RETRIEVAL:
        # check ppl calculation
        # ppl = torch.exp(torch.stack(nlls).mean()).cpu()
        ppl = torch.exp(torch.stack(nlls).sum() / seq_len).cpu()
        assert np.abs(ppl - ppl_over_logits) < 1e-3, f"{ppl:.3f}, {ppl_over_logits:.3f}"

    return ppl_over_logits


if __name__ == "__main__":
    main()