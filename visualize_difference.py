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
    print(data_name)
    for qid in os.listdir(f"./differences/{data_name}"):
        if qid.endswith(".pkl"):
            qid = qid.split("_")[1].replace(".pkl", "")
            df_before = pd.read_pickle(f"./differences/{data_name}/before_{qid}.pkl")
            df_after = pd.read_pickle(f"./differences/{data_name}/after_{qid}.pkl")

            df_before_non_rel = df_before[df_before['relevant'] != True].head(n = 50)
            df_after_non_rel = df_after[df_after['relevant'] != True].head(n = 50)
            accumulate_before.extend(df_before_non_rel['cross_encoder_score'])
            accumulate_after.extend(df_after_non_rel['cross_encoder_score'])

    sns.histplot(accumulate_before,color="skyblue", label="Before Domain Adaptation", kde=True)
    sns.histplot(accumulate_after, color="red", label="After Domain Adaptation", kde=True)
    plt.title(f'{data_name}')
    plt.legend() 
    plt.show()
    plt.savefig(f"difference_plots/{data_name}.png")
    
if __name__ == "__main__":
    main()