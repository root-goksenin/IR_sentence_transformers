import torch 
from TensorboardWriters import TensorBoardAttentionWriter
from utils import load_json
import click
from utils import concat_title_and_body

base_name = "GPL/msmarco-distilbert-margin-mse"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
@click.command()
@click.argument('data_name')       
def generate_logs(data_name):
    adapted = f"GPL/{data_name}-msmarco-distilbert-gpl"
    writer = TensorBoardAttentionWriter(data_name, base_name, adapted, device)
    diffs = load_json(f"differences/{data_name}.json")
    # Write query
    pos, _ = list(diffs.keys())[0], list(diffs.keys())[1]
    writer.write(writer.queries[pos], pos)
    for did in writer.qrels[pos]:
        writer.write(concat_title_and_body(did, writer.corpus), did)
    writer.write(writer.queries[pos], pos, types = "model")
    for did in writer.qrels[pos]:
        writer.write(concat_title_and_body(did, writer.corpus), did, types = "model")
if __name__ == "__main__":
    generate_logs()

