# We need the pooling output from [PAD] [PAD] query
# We need the pooling output from [PAD] [PAD] document
from SentenceTransformerWrapper import SentenceTransformerWrapper 
from torch import nn 
import torch 
from sentence_transformers.util import dot_score 
import sys
sys.path.append("../master_thesis_ai")
from gpl_improved.utils import load_sbert
from captum.attr import IntegratedGradients
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class IRWrapper(nn.Module):
    def __init__(self, model):
        super(IRWrapper, self).__init__()
        self.model = model

    def forward_with_features(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self.bert_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if self.bert_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})
        
        return self.pooler(features)
    
    def forward(self, features_query, features_doc):
        q_emb = self.forward_with_features(features_query)
        doc_emb = self.forward_with_features(features_doc)
        return dot_score(q_emb, doc_emb)


if __name__ == "__main__":
    base = "GPL/msmarco-distilbert-margin-mse"
    sentence_transformer_model = load_sbert(base)
    model = IRWrapper(sentence_transformer_model)
    parts = SentenceTransformerWrapper(sentence_transformer_model, device)
    
    query = ["Hello How u doing?"]
    doc = "I am doing fine thanks."
    
    # query_input, query_ref = {"input_ids": query_input_ids, "attention_mask": query_attention_mask},
    #                          {"input_ids": query_ref_input_ids, "attention_mask": query_ref_attention_mask}
   
    # doc_input, doc_ref = {"input_ids": doc_input_ids, "attention_mask": doc_attention_mask},
    #                     {"input_ids": doc_ref_input_ids, "attention_mask": doc_ref_attention_mask}
    # query_input, query_ref = 
    # doc_input, doc_ref = 
    
    # ig = IntegratedGradients(model)
    # attr, delta = ig.attribute(inputs = (query_input, doc_input), baselines = (query_ref, doc_ref), return_convergance_delta = True)

    print(parts.return_text_and_base_features(query))

    
