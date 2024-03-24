# We need the pooling output from [PAD] [PAD] query
# We need the pooling output from [PAD] [PAD] document
from SentenceTransformerWrapper import SentenceTransformerWrapper 
from torch import nn 
import torch 
from sentence_transformers.util import dot_score 
import sys
sys.path.append("../master_thesis_ai")
from gpl_improved.utils import load_sbert
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from IntegratedGradientsTransformer import IRWrapperDoc, IRWrapperQuery
from beir.datasets.data_loader import GenericDataLoader
from utils import beir_path, concat_title_and_body, load_json
from captum.attr import visualization as viz
import os 
import click 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions



    
def generate_attributions(query, doc, model, for_query = False):
    
    sentence_transformer_model_query = load_sbert(model)
    parts_query = SentenceTransformerWrapper(sentence_transformer_model_query, device)
    sentence_transformer_model_doc = load_sbert(model)
    parts_doc = SentenceTransformerWrapper(sentence_transformer_model_doc, device)
    
    
    query_input, query_ref, query_tokens = parts_query.return_text_and_base_features(query) 
    doc_input, doc_ref, doc_tokens =  parts_doc.return_text_and_base_features(doc) 
    
    # Generate query and ref
    query_input_ids, query_attention_mask = query_input['input_ids'], query_input['attention_mask']
    query_ref_ids, _ = query_ref['input_ids'], query_ref['attention_mask']
    
    # Generate doc and ref
    doc_input_ids, doc_attention_mask = doc_input['input_ids'], doc_input['attention_mask']
    doc_ref_ids, _ = doc_ref['input_ids'], doc_ref['attention_mask']
    
    
    
    if for_query:
        model_q = IRWrapperQuery(parts_query.bert_model, parts_doc.bert_model, parts_query.pooler, doc_input_ids, doc_attention_mask)
        # Get the query model embeddings
        ig_q = LayerIntegratedGradients(model_q, model_q.query_model.embeddings)
        
        return  query_tokens, ig_q.attribute(inputs = query_input_ids, 
                            baselines = query_ref_ids,
                            internal_batch_size = 1,
                            additional_forward_args = (query_attention_mask),
                            )
    else:
        model_d = IRWrapperDoc(parts_query.bert_model, parts_doc.bert_model, parts_query.pooler, query_input_ids, query_attention_mask)
        # Get the doc model embeddings.
        ig_d = LayerIntegratedGradients(model_d, model_d.doc_model.embeddings)
        
        return  doc_tokens, ig_d.attribute(inputs = doc_input_ids, 
                            baselines = doc_ref_ids,
                            internal_batch_size = 1,
                            additional_forward_args = (doc_attention_mask),
                            )
        
      
def visualize(attr, tokens, title):    
    return viz.VisualizationDataRecord(
                        attr,
                        0,
                        0,
                        0,
                        title,
                        attr.sum(),       
                        tokens,
                        0)
def facade(model_before, model_after, query, doc, data_name, qid, title_prefix):

    query_tokens, q_attr = generate_attributions(query, doc, model_before, for_query = True)
    q_attr  = summarize_attributions(q_attr)
    doc_tokens, d_attr = generate_attributions(query, doc, model_before, for_query = False)
    doc_attr =  summarize_attributions(d_attr)
    
    query_vis = visualize(q_attr, query_tokens, title = title_prefix + "Before Query")
    doc_vis = visualize(doc_attr, doc_tokens,  title =  title_prefix + "Before Doc")
    html = viz.visualize_text([query_vis, doc_vis])
    _write_html(f"attributions/{data_name}/{qid}/before_attr.html", html.data)
    
    query_tokens, q_attr = generate_attributions(query, doc, model_after, for_query = True)
    q_attr  = summarize_attributions(q_attr)
    doc_tokens, d_attr = generate_attributions(query, doc, model_after, for_query = False)
    doc_attr =  summarize_attributions(d_attr)

    query_vis_a = visualize(q_attr, query_tokens, title = title_prefix + "After Query")
    doc_vis_a = visualize(doc_attr, doc_tokens, title = title_prefix + "After Doc")
    html = viz.visualize_text([query_vis, doc_vis])
    _write_html(f"attributions/{data_name}/{qid}/after_attr.html", html.data)
    
    html = viz.visualize_text([query_vis, doc_vis, query_vis_a, doc_vis_a])
    _write_html(f"attributions/{data_name}/{qid}/attr.html", html.data)
    
def _write_html(path, html):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w') as file:
        file.write(html)


@click.command()
@click.argument("data_name")
def main(data_name):
    base = "GPL/msmarco-distilbert-margin-mse"  
    adapted = f"GPL/{data_name}-msmarco-distilbert-gpl" 
    corpus, queries, qrels = GenericDataLoader(beir_path.format(data_name)).load("test")

    for qid, key in load_json(f"differences/{data_name}.json").items():
        query = [queries[qid]]
        
        doc = [concat_title_and_body(list(qrels[qid].keys())[0], corpus)]
        facade(model_before = base,
            model_after = adapted,
            query= query,
            doc = doc,
            data_name = data_name,
            qid = qid, 
            title_prefix = "Worse " if key <0 else "Better ")
    
if __name__ == "__main__":
    main()
    

