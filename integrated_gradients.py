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
from beir.datasets.data_loader import GenericDataLoader
from utils import beir_path, concat_title_and_body
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class IRWrapperQuery(nn.Module):
     def __init__(self, query_model, doc_model, pooler, doc_input_ids, doc_attention_mask):
        super(IRWrapperQuery, self).__init__()
        self.pooler = pooler
        self.query_model = query_model
        self.doc_model = doc_model
        self.d = doc_input_ids
        self.d_att = doc_attention_mask

     def forward_with_features(self, features, model):
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        trans_features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            trans_features.update({"all_layer_embeddings": hidden_states})
        
        return self.pooler(trans_features)['sentence_embedding']

     def forward(self, query_input_ids, query_attention_mask):

        features_doc = {'input_ids': self.d, 'attention_mask' : self.d_att}
        features_query = {'input_ids': query_input_ids, 'attention_mask': query_attention_mask}
    
        q_emb = self.forward_with_features(features_query, self.query_model)
        doc_emb = self.forward_with_features(features_doc, self.doc_model)
        score = dot_score(doc_emb, q_emb)
        return score

class IRWrapperDoc(nn.Module):
    def __init__(self, query_model, doc_model, pooler, query_input_ids, query_attention_mask):
        super(IRWrapperDoc, self).__init__()
        self.pooler = pooler
        self.query_model = query_model
        self.doc_model = doc_model
        self.q = query_input_ids
        self.q_att = query_attention_mask

    def forward_with_features(self, features, model):
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        trans_features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            trans_features.update({"all_layer_embeddings": hidden_states})
        
        return self.pooler(trans_features)['sentence_embedding']
    
    def forward(self, doc_input_ids , doc_attention_mask):
         # sourcery skip: inline-immediately-returned-variable
        
        features_query = {'input_ids': self.q, 'attention_mask': self.q_att}
        features_doc = {'input_ids': doc_input_ids, 'attention_mask' : doc_attention_mask}
        
        q_emb = self.forward_with_features(features_query, self.query_model)
        doc_emb = self.forward_with_features(features_doc, self.doc_model)
        score = dot_score(q_emb, doc_emb)
        return score



class IRWrapper(nn.Module):
    def __init__(self, query_model, doc_model, pooler):
        super(IRWrapper, self).__init__()
        self.pooler = pooler
        self.query_model = query_model
        self.doc_model = doc_model

    def forward_with_features(self, features, model):
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        trans_features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            trans_features.update({"all_layer_embeddings": hidden_states})
        
        return self.pooler(trans_features)['sentence_embedding']
    
    def forward(self, query_input_ids, doc_input_ids ,query_attention_mask, doc_attention_mask):
         # sourcery skip: inline-immediately-returned-variable
        
        features_query = {'input_ids': query_input_ids, 'attention_mask': query_attention_mask}
        features_doc = {'input_ids': doc_input_ids, 'attention_mask' : doc_attention_mask}
        q_emb = self.forward_with_features(features_query, self.query_model)
        doc_emb = self.forward_with_features(features_doc, self.doc_model)
        score = dot_score(q_emb, doc_emb)
        return score
    
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

if __name__ == "__main__":
    base = "GPL/msmarco-distilbert-margin-mse"
    
    sentence_transformer_model = load_sbert(base)
    parts = SentenceTransformerWrapper(sentence_transformer_model, device)
    
    
    data_name = "scifact"
    corpus, queries, qrels = GenericDataLoader(beir_path.format(data_name)).load("test")

    query = [queries["99"]]
    print(qrels["99"])
    doc = [concat_title_and_body("18810195", corpus)]
    
    query_input, query_ref = parts.return_text_and_base_features(query) 
    doc_input, doc_ref =  parts.return_text_and_base_features(doc) 
    query_input_ids, query_attention_mask = query_input['input_ids'], query_input['attention_mask']
    doc_input_ids, doc_attention_mask = doc_input['input_ids'], doc_input['attention_mask']
    query_ref_ids, query_ref_att = query_ref['input_ids'], query_ref['attention_mask']
    doc_ref_ids, doc_ref_att = doc_ref['input_ids'], doc_ref['attention_mask']    
    print(parts.decode(query_input_ids))
 
    # model_d = IRWrapperDoc(parts_query.bert_model,parts_doc.bert_model, parts_query.pooler, query_input_ids.to(device), query_attention_mask.to(device))
    # ig_d = LayerIntegratedGradients(model_d, parts_doc.bert_model.embeddings)
    # inputs = doc_input_ids.to(device)
    # baselines = doc_ref_ids.to(device)
    # attr_d = ig_d.attribute(inputs = inputs, 
    #                     baselines = baselines,
    #                     additional_forward_args = (doc_attention_mask.to(device)),
    #                     internal_batch_size = 1,
    #                     )
    
    
    # inputs = query_input_ids.to(device)
    # baselines = query_ref_ids.to(device)
    # model_q = IRWrapperQuery(parts_query.bert_model, parts_doc.bert_model, parts_query.pooler, doc_input_ids.to(device), doc_attention_mask.to(device))
    # ig_q = LayerIntegratedGradients(model_q, model_q.query_model.embeddings)
    # attr_q = ig_q.attribute(inputs = inputs, 
    #                     baselines = baselines,
    #                     internal_batch_size = 1,
    #                     additional_forward_args = (query_attention_mask.to(device)),
    #                     )
 
    # attributions_start_sum_q = summarize_attributions(attr_q)
    # attributions_start_sum_d = summarize_attributions(attr_d)

    # print(attributions_start_sum_q)
    # print(attributions_start_sum_d)
    
    
    
    sentence_transformer_model_query = load_sbert(base)
    parts_query = SentenceTransformerWrapper(sentence_transformer_model_query, device)
    sentence_transformer_model_doc = load_sbert(base)
    parts_doc = SentenceTransformerWrapper(sentence_transformer_model_doc, device)
    
    inputs = (query_input_ids.to(device), doc_input_ids.to(device))
    baselines = (query_ref_ids.to(device), doc_ref_ids.to(device))
    model = IRWrapper(parts_query.bert_model, parts_doc.bert_model, parts_query.pooler)
    ig = LayerIntegratedGradients(model, [model.query_model.embeddings, model.doc_model.embeddings])
    query_attr, doc_attr = ig.attribute(inputs = inputs, 
                        baselines = baselines,
                        internal_batch_size = 1,
                        additional_forward_args = (query_attention_mask.to(device), doc_attention_mask.to(device)),
                        n_steps = 700
                        )
    print(summarize_attributions(query_attr), summarize_attributions(doc_attr))
    
    
    base = "GPL/scifact-msmarco-distilbert-gpl"

    sentence_transformer_model_query = load_sbert(base)
    parts_query = SentenceTransformerWrapper(sentence_transformer_model_query, device)
    sentence_transformer_model_doc = load_sbert(base)
    parts_doc = SentenceTransformerWrapper(sentence_transformer_model_doc, device)
    
    inputs = (query_input_ids.to(device), doc_input_ids.to(device))
    baselines = (query_ref_ids.to(device), doc_ref_ids.to(device))
    model = IRWrapper(parts_query.bert_model, parts_doc.bert_model, parts_query.pooler)
    ig = LayerIntegratedGradients(model, [model.query_model.embeddings, model.doc_model.embeddings])
    query_attr, doc_attr = ig.attribute(inputs = inputs, 
                        baselines = baselines,
                        internal_batch_size = 1,
                        n_steps = 700,
                        additional_forward_args = (query_attention_mask.to(device), doc_attention_mask.to(device)),
                        )
    print(summarize_attributions(query_attr), summarize_attributions(doc_attr))
    

    
 
    

