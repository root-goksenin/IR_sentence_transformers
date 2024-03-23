
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers.util import dot_score
from beir.retrieval import models

class SentenceTransformerWrapper:
    def __init__(self, model, device):
        self.model = model 
        # DistilBertModel from huggingface, put it to gpu
        self.bert_model = model._first_module().auto_model
        self.bert_model.to(device)
        self.bert_model.eval()
        self.bert_model.zero_grad()
        self.tokenizer = model._first_module().tokenizer 
        # This is a wrapper function that tokenizer inputs.
        self.bert_tokenizer = model._first_module().tokenize
        self.ref_token_id = self.tokenizer.pad_token_id # A token used for generating token reference
        self.sep_token_id = self.tokenizer.sep_token_id # A token used for adding it to the end of the text.
        self.cls_token_id = self.tokenizer.cls_token_id # A token used for adding it to the beggining of the text.
        self.searcher = DenseRetrievalExactSearch(self._from_sbert_to_beir(), batch_size=256, corpus_chunk_size=256*1000) 
    
    def _from_sbert_to_beir(self):
      retriever = models.SentenceBERT(sep=" ")
      retriever.q_model = self.model
      retriever.doc_model = self.model
      return retriever
                
    
    def _produce_embedding(self, input_text):
        input_texts = [input_text]
        return self.model.encode(input_texts)[0]

    def calculate_sim(self, query, doc):
        return dot_score(self._produce_embedding(query),self._produce_embedding(doc))
    
    def return_hard_negatives(self, qid, query, corpus, qrels):
        hard_negatives = []
        dummy_query = {qid : query, "AABSEHFFDWD": query}
        returned = self.searcher.search(corpus, dummy_query,
               top_k = 1000 + len(qrels[qid]), 
               score_function = "dot",
               return_sorted = True)
        for did in returned[qid]:
            if did not in qrels[qid]: 
                hard_negatives.append(did)
        return hard_negatives
        