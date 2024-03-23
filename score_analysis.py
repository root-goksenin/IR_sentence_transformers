
import os 
import json 
from utils import load_json
import click 

def find_difference(before, after):
    new_dic = {}
    for key in before: 
        diff = after[key]["ndcg@10"] - before[key]["ndcg@10"]
        try: 
            key = int(key)
            key += 1 
        except:
            print("no int")
        new_dic[key] = diff
    return sorted(new_dic.items(), key = lambda x: x[1], reverse = True)



@click.command()
@click.argument("data_name")
def main(data_name):
    if not os.path.exists(f"differences/{data_name}.json", "w"):
        zero_shot_beir_data = f"../master_thesis_ai/zero_shot_results/{data_name}/GPL/msmarco-distilbert-margin-mse/results_query_level.json"
        domain_adapted_beir_data = f"../master_thesis_ai/zero_shot_results/{data_name}/GPL/{data_name}-msmarco-distilbert-gpl/results_query_level.json"


        before, after = load_json(zero_shot_beir_data), load_json(domain_adapted_beir_data)
        diff = find_difference(before, after)
                
        with open(f"differences/{data_name}.json", "w") as file:
            json.dump(dict([diff[0], diff[-1]]), file)
