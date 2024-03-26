import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
  
sns.set_theme(style="darkgrid")


accumulate_before = []
accumulate_after = []
for qid in os.listdir("."):
    if qid.endswith(".pkl"):
        qid = qid.split("_")[1].replace(".pkl", "")
        df_before = pd.read_pickle(f"before_{qid}.pkl")
        df_after = pd.read_pickle(f"after_{qid}.pkl")

        df_before_non_rel = df_before[df_before['relevant'] != True].head(n = 100)
        df_after_non_rel = df_after[df_after['relevant'] != True].head(n = 100)
        accumulate_before.extend(df_before_non_rel['cross_encoder_score'])
        accumulate_after.extend(df_after_non_rel['cross_encoder_score'])

sns.histplot(accumulate_before,color="skyblue", label="Before Domain Adaptation", kde=True)
sns.histplot(accumulate_after, color="red", label="After Domain Adaptation", kde=True)

plt.legend() 
plt.show()
plt.savefig("../../difference_plots/scidocs.png")
    