import sys
sys.path.append("../master_thesis_ai")
from gpl_improved.utils import load_sbert
from beir.datasets.data_loader import GenericDataLoader
from utils import beir_path, concat_title_and_body
from SentenceTransformerWrapper import SentenceTransformerWrapper
import os 
import tensorflow as tf 
from tensorboard.plugins import projector
import numpy as np
from bertviz import head_view, model_view



class TensorBoardAttentionWriter:
    
    def __init__(self, data_name, model_before, model_after, device):
        self.data_name = data_name
        self.corpus, self.queries, self.qrels = GenericDataLoader(beir_path.format(data_name)).load("test")
        model_before = load_sbert(model_before)
        model_after = load_sbert(model_after)
        self.wrapped_before = SentenceTransformerWrapper(model_before, device)
        self.wrapped_after = SentenceTransformerWrapper(model_after, device)
         
    def _write_html(self, path, html):
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, 'w') as file:
            file.write(html)


    def write(self, text, id, types = 'head'):
        if types == "head":
            self.write_head(text, id)
        else:
            self.write_model(text, id)
            
    def write_head(self, text, id):        
        attention,tokens = self.wrapped_before.return_attention(text)
        html_head_view = head_view(attention, tokens, html_action='return',include_layers = [5])
        self._write_html(f"attentions/{self.data_name}/before/head_view_{id}.html" , html_head_view.data)
       
        attention,tokens = self.wrapped_after.return_attention(text)
        html_head_view = head_view(attention, tokens, html_action='return', include_layers = [5])
        self._write_html(f"attentions/{self.data_name}/after/head_view_{id}.html" , html_head_view.data)

    def write_model(self, text, id):        
        attention,tokens = self.wrapped_before.return_attention(text)
        html_head_view = model_view(attention, tokens, html_action='return', include_layers = [5])
        self._write_html(f"attentions/{self.data_name}/before/model_view_{id}.html" , html_head_view.data)
       
        attention,tokens = self.wrapped_after.return_attention(text)
        html_head_view = model_view(attention, tokens, html_action='return', include_layers = [5])
        self._write_html(f"attentions/{self.data_name}/after/model_view_{id}.html" , html_head_view.data)


class TensorBoardProjectorWriter:
    def __init__(self, data_name, model_before, model_after, device):
        self.data_name = data_name
        self.corpus, self.queries, self.qrels = GenericDataLoader(beir_path.format(data_name)).load("test")
        model_before = load_sbert(model_before)
        model_after = load_sbert(model_after)
        self.wrapped_before = SentenceTransformerWrapper(model_before, device)
        self.wrapped_after = SentenceTransformerWrapper(model_after, device)
        
        
    def _generate_embeddings(self, qid, hard_negatives,embedder):
        query_embedding = np.expand_dims(embedder._produce_embedding(self.queries[qid]), axis = 0)
        
        negative_embeddings = np.zeros((len(hard_negatives), 768))
        for id, did in enumerate(hard_negatives):
            doc = concat_title_and_body(did, self.corpus)
            negative_embeddings[id] = embedder._produce_embedding(doc)
        
        positive_embeddings = np.zeros((len(self.qrels[qid]), 768))
        for id, did in enumerate(self.qrels[qid]):
            doc = concat_title_and_body(did, self.corpus)
            positive_embeddings[id] = embedder._produce_embedding(doc)
        
        return np.concatenate((query_embedding, negative_embeddings, positive_embeddings))
            
    def write(self, qid):
        log_dir = f"logs/{self.data_name}/{qid}"
        # Get the query
        query = self.queries[qid]
        hard_negatives = self.wrapped_before.return_hard_negatives(
                            qid, query,  self.corpus, self.qrels)
        self._generate_labels(hard_negatives, qid, log_dir)
        before = self._generate_embeddings(qid, hard_negatives, self.wrapped_before)
        after = self._generate_embeddings(qid, hard_negatives, self.wrapped_after)
        # Save the weights we want to analyze as a variable. Note that the first
        # value represents any unknown word, which is not in the metadata, here
        # we will remove this value.
        weights_before = tf.Variable(before, name = "before")
        weights_after = tf.Variable(after, name = "after")
        # Create a checkpoint from embedding, the filename and key are the
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(before=weights_before, after=weights_after)
        checkpoint.save(os.path.join(log_dir, "var.ckpt"))
        # Set up config.
        config = projector.ProjectorConfig()
        embeddings_before = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
        embeddings_before.tensor_name = "before/.ATTRIBUTES/VARIABLE_VALUE"
        embeddings_before.metadata_path = 'metadata.tsv'
        # Add After Embeddings
        embeddings_after = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
        embeddings_after.tensor_name = "after/.ATTRIBUTES/VARIABLE_VALUE"
        embeddings_after.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir, config)


    def _generate_labels(self, negative_ids, qid, log_dir):
        labels = [] 
        labels.append(("Query", self.queries[qid]))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        for did in negative_ids:
            doc = concat_title_and_body(did, self.corpus)
            labels.append(("Non Related", doc))
        for did,rel in self.qrels[qid].items():
            doc = concat_title_and_body(did, self.corpus)
            labels.append((f"Relatedness {rel}", doc))
                # Save Labels separately on a line-by-line manner.
        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            f.write("Label\tText\n")
            for label, text in labels:
                f.write(f"{label}\t{text}\n")