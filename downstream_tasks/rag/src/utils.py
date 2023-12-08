import os 


def get_model_name(model_path):

    if os.path.isdir(model_path):
        model_name, seg_type = model_path.strip('/').split('/')[-2:]

    if model_name == 'bert-base-cased':
        name = 'bert'
    elif model_name == 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':
        name = 'PubMedBERT'
    else:
        name = model_name

    assert '-best' in seg_type
    seg_type = seg_type.replace('-best', '')

    return name, seg_type


def get_index_name(retriever_model_path, ds_path, num_entries, return_emb_name=False):
    model_name, model_seq_type = get_model_name(retriever_model_path)

    datastore= ds_path.strip('/').split('/')[-1] # chunks sent or seg

    base_name = f"{model_name}_{model_seq_type}-{datastore}_{num_entries}"

    index_name = f"index-{base_name}.faiss"
    embed_name = f"embed-{base_name}.npy"

    if return_emb_name: return embed_name

    return index_name


def get_outfile_path(args):

    lm_name = args.lm_model_name_or_path.strip('/').split('/')[-1]

    model_name, seg_type = get_model_name(args.retriever_model_name_or_path)

    test_set = args.eval_data_path.strip('/').split('_')[-1]
    datastore= args.datastore_path.strip('/').split('_')[-1]

    subfolder = f'lm_{lm_name}-test_{test_set}'

    out_dir = os.path.join('results', model_name, 'PPL', subfolder)
    os.makedirs(out_dir, exist_ok=True)

    config = f's{args.stride}l{args.max_length}k{args.topK}'

    outfile = os.path.join(out_dir, f'ds_{datastore}-model_{seg_type}-{config}.txt')

    return outfile
    
