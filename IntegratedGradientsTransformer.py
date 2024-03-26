from SentenceTransformerWrapper import SentenceTransformerWrapper 
from torch import nn 
import torch 
from sentence_transformers.util import dot_score, cos_sim
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
        # We need both query model and the doc model for Integrated gradients to work. 
        self.query_model = query_model
        self.doc_model = doc_model
        # Give the document already. Now we need to see how query interacts with the document
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
    
        # Get the query features, and pass it to the model
        q_emb = self.forward_with_features(features_query, self.query_model)
        # Get the document features, and pass it to the model.
        doc_emb = self.forward_with_features(features_doc, self.doc_model)
        score = dot_score(doc_emb, q_emb)
        return score.diagonal()
    
    
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
        return score.diagonal()
    
    
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