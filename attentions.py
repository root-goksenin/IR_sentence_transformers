import torch 
from SentenceTransformerWrapper import SentenceTransformerWrapper
from TensorboardWriters import TensorBoardProjectorWriter
from utils import load_json
import click

base_name = "GPL/msmarco-distilbert-margin-mse"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
@click.command()
@click.argument('data_name')       
def generate_logs(data_name):
    adapted = f"GPL/{data_name}-msmarco-distilbert-gpl"
    writer = TensorBoardProjectorWriter(data_name, base_name, adapted, device)
    diffs = load_json(f"differences/{data_name}.json")
    pos, neg = list(diffs.keys())[0], list(diffs.keys())[1]
    writer.write(pos)
    writer.write(neg)
    
if __name__ == "__main__":
    generate_logs()

