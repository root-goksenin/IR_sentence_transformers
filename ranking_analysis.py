
from utils import load_json, beir_path
from SentenceTransformerWrapper import SentenceTransformerWrapper
import click 
from beir.datasets.data_loader import GenericDataLoader
import torch 
import sys
sys.path.append("../master_thesis_ai")
from gpl_improved.utils import load_sbert
import pandas as pd 
import os
import pandas as pd 
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer
from utils import concat_title_and_body

teacher = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
retokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2") 


def retokenize(texts, retokenizer):
    ## We did this retokenization for two reasons:
    ### (1) Setting the max_seq_length;
    ### (2) We cannot simply use CrossEncoder(cross_encoder, max_length=max_seq_length),
    ##### since the max_seq_length will then be reflected on the concatenated sequence,
    ##### rather than the two sequences independently
    texts = list(map(lambda text: text.strip(), texts))
    features = retokenizer(
        texts,
        padding=True,
        truncation="longest_first",
        return_tensors="pt",
        max_length=350
    )
    return retokenizer.batch_decode(
        features["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
def predict(query, doc):
    with torch.no_grad():
            query, doc = [retokenize(texts, retokenizer) for texts in [query, doc]]
            scores = teacher.predict(
                list(zip(query, doc)), show_progress_bar=False, convert_to_tensor = True
            )
    return scores.tolist()
            
    
def predict_dataset_relevance(df, queries, corpus):
    qid = df['qid'][0]
    dids = df['did']
    query = [queries[qid]] * len(dids)
    docs = [concat_title_and_body(did, corpus) for did in dids]
    
    return predict(query, docs)
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_ranks(search_set, list):
    ranks = []
    for did in search_set:
        ranks.append(list.index(did))
    return ranks
        
        
        

def return_documents_pre_computed(model, corpus, queries, qrels, k = 100, score_function = "dot"):
    return_dict = model.searcher.search( 
               corpus,
               queries,
               top_k = k, 
               score_function = score_function,
               return_sorted = True)
    datasets = {}
    for qid in qrels:
        ranked_documents = list(dict(sorted(return_dict[qid].items(), key=lambda item: item[1], reverse = True)).keys())
        dataset = pd.DataFrame(columns=['qid','did', 'rank', 'relevant'])
        for i in range(len(ranked_documents)):
            dataset.loc[i] = [qid] + [ranked_documents[i]] + [i] + [ranked_documents[i] in qrels[qid].keys()]
        dataset['cross_encoder_score'] = predict_dataset_relevance(dataset, queries, corpus)
        datasets[qid] = dataset
    return datasets
        
def return_documents(model,corpus, queries, qrels, qid):
    
    relevant_docs =  set(qrels[qid].keys())
    before_documents = model.return_top_k(qid, queries, corpus, qrels, top_k = 100)
    dataset = pd.DataFrame(columns=['qid','did', 'rank', 'relevant'])
    for i in range(len(before_documents)):
        dataset.loc[i] = [qid] + [before_documents[i]] + [i] + [before_documents[i] in relevant_docs]
    dataset['cross_encoder_score'] = predict_dataset_relevance(dataset, queries, corpus)
    return dataset


def write_to_disk(datasets, save_path):
    for qid, dataset in datasets.items():
        dataset.to_pickle(save_path + f"{qid}.pkl")        


@click.command()
@click.argument("data_name")
def main(data_name):

    os.makedirs(f"differences/{data_name}", exist_ok = True)
    base = "GPL/msmarco-distilbert-margin-mse"  
    adapted = f"GPL/{data_name}-msmarco-distilbert-gpl" 
    corpus, queries, qrels = GenericDataLoader(beir_path.format(data_name)).load("test")
    before_model = load_sbert(base)
    before_model = SentenceTransformerWrapper(before_model, device)
    after_model = load_sbert(adapted)
    after_model = SentenceTransformerWrapper(after_model, device)
    
    hard_negative_miner_1 = SentenceTransformerWrapper(SentenceTransformer("msmarco-distilbert-base-v3"), device)
    hard_negative_miner_2 = SentenceTransformerWrapper(SentenceTransformer("msmarco-MiniLM-L-6-v3"), device)
            
    before_datasets = return_documents_pre_computed(before_model, corpus, queries, qrels, score_function = "dot")
    after_datasets = return_documents_pre_computed(after_model, corpus, queries, qrels, score_function = "dot")
    
    hard_negative_miner_1_datasets = return_documents_pre_computed(hard_negative_miner_1, corpus, queries, qrels, score_function = "cos_sim")
    hard_negative_miner_2_datasets = return_documents_pre_computed(hard_negative_miner_2, corpus, queries, qrels, score_function = "cos_sim")
    write_to_disk(before_datasets, save_path = f"differences/{data_name}/before_" )
    write_to_disk(after_datasets, save_path = f"differences/{data_name}/after_")
    write_to_disk(hard_negative_miner_1_datasets, save_path = f"differences/{data_name}/hard_negative_miner_1_" )
    write_to_disk(hard_negative_miner_2_datasets, save_path = f"differences/{data_name}/hard_negative_miner_2_")
        
if __name__ == "__main__":
    main()