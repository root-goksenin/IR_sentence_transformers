import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import re

for qid in os.listdir("."):
    if qid.endswith(".pkl"):
        qid = re.findall(r'\d+', qid)[0]
        df_before = pd.read_pickle(f"before_{qid}.pkl")
        df_after = pd.read_pickle(f"after_{qid}.pkl")

        df_before_non_rel = df_before[df_before['relevant'] != True].head(n = 100)
        df_after_non_rel = df_after[df_after['relevant'] != True].head(n = 100)
        plt.figure()
        plt.hist(df_before_non_rel['cross_encoder_score'], label = "Base model top 100 hard negatives", alpha  = 0.5)
        plt.hist(df_after_non_rel['cross_encoder_score'], label = "Domain adapted model top 100 hard negatives", alpha = 0.5)
        plt.legend()
        plt.title(f"Hard negative document's query relevancy scores for QID {qid}")
        plt.savefig(f"{qid}.png")
    