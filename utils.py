import json 

beir_path = "../master_thesis_ai/beir_data/{}"
def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def concat_title_and_body(did: str, corpus):
  assert type(did) == str
  document = []
  title = corpus[did]["title"].strip()
  body = corpus[did]["text"].strip()
  if len(title):
      document.append(title)
  if len(body):
      document.append(body)
  return " ".join(document)

