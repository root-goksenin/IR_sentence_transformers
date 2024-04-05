import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import click 
  
  
sns.set_theme(style="darkgrid")

@click.command()
@click.argument("data_name")
def main(data_name):
    accumulate_before = []
    accumulate_after = []
    accumulate_hm1 = []
    accumulate_hm2 = []
    print(data_name)
    for qid in os.listdir(f"./differences/{data_name}"):
        if qid.endswith(".pkl"):
            qid = qid.split("_")[-1].replace(".pkl", "")
            df_before = pd.read_pickle(f"./differences/{data_name}/before_{qid}.pkl")
            df_after = pd.read_pickle(f"./differences/{data_name}/after_{qid}.pkl")
            df_hm_1 = pd.read_pickle(f"./differences/{data_name}/hard_negative_miner_1_{qid}.pkl")
            df_hm_2 = pd.read_pickle(f"./differences/{data_name}/hard_negative_miner_2_{qid}.pkl")
            df_before_non_rel = df_before[df_before['relevant'] != True].head(n = 50)
            df_after_non_rel = df_after[df_after['relevant'] != True].head(n = 50)
            df_hm1_non_rel = df_hm_1[df_hm_1['relevant'] != True].head(n = 50)
            df_hm2_non_rel = df_hm_2[df_hm_2['relevant'] != True].head(n = 50)
            accumulate_before.extend(df_before_non_rel['cross_encoder_score'])
            accumulate_after.extend(df_after_non_rel['cross_encoder_score'])
            accumulate_hm1.extend(df_hm1_non_rel['cross_encoder_score'])
            accumulate_hm2.extend(df_hm2_non_rel['cross_encoder_score'])

    sns.histplot(accumulate_before,color=sns.xkcd_rgb["royal"], label="Before Domain Adaptation", kde=True, alpha = 0.3)
    sns.histplot(accumulate_after, color="red", label="After Domain Adaptation", kde=True, alpha = 0.6)
    sns.histplot(accumulate_hm1,color=sns.xkcd_rgb["hot pink"], label="Hard Negative Miner 1", kde=True, alpha = 0.3)
    sns.histplot(accumulate_hm2, color=sns.xkcd_rgb["jungle green"], label="Hard Negative Miner 2", kde=True, alpha = 0.3)
    
    plt.title(f'{data_name}')
    plt.legend() 
    plt.show()
    plt.savefig(f"difference_plots/{data_name}.png")
    
if __name__ == "__main__":
    main()