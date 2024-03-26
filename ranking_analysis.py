
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
from sentence_transformers import CrossEncoder
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
        
        
        
# TODO!
# Make it so that documents are already indexed, and multiple queries can be used
# with indexed documents. Because, we know that documents wont change.

def return_documents(model,corpus, queries, qrels, qid):
    
    relevant_docs =  set(qrels[qid].keys())
    before_documents = model.return_top_k(qid, queries, corpus, qrels, top_k = 100)
    dataset = pd.DataFrame(columns=['qid','did', 'rank', 'relevant'])
    for i in range(len(before_documents)):
        dataset.loc[i] = [qid] + [before_documents[i]] + [i] + [before_documents[i] in relevant_docs]
    
    dataset['cross_encoder_score'] = predict_dataset_relevance(dataset, queries, corpus)
    return dataset


@click.command()
@click.argument("data_name")
def main(data_name):

    os.makedirs(f"differences/{data_name}", exist_ok = True)
    diff = load_json(f"differences/{data_name}.json")
    base = "GPL/msmarco-distilbert-margin-mse"  
    adapted = f"GPL/{data_name}-msmarco-distilbert-gpl" 
    corpus, queries, qrels = GenericDataLoader(beir_path.format(data_name)).load("test")
    before_model = load_sbert(base)
    before_model = SentenceTransformerWrapper(before_model, device)
    after_model = load_sbert(adapted)
    after_model = SentenceTransformerWrapper(after_model, device)
            
    for qid in diff.keys():
        before = return_documents(before_model,
                        corpus, queries, qrels, qid)
        before.to_pickle(f"differences/{data_name}/before_{qid}.pkl")
        after = return_documents(after_model,
                        corpus, queries, qrels, qid)
        after.to_pickle(f"differences/{data_name}/after_{qid}.pkl")        
    
        
if __name__ == "__main__":
    main()