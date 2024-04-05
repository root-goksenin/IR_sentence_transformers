import tensorflow as tf
import matplotlib.pyplot as plt 
import glob
import re 
import numpy as np

pattern = r'(?<=_)\d+(?=_gpl)'


def get_scalar_run_tensorboard(tag, filepath):
    values,steps = [],[]
    for e in tf.compat.v1.train.summary_iterator(filepath):
        if len(e.summary.value)>0: #Skip first empty element
            if e.summary.value[0].tag==tag:
                tensor = (e.summary.value[0].simple_value)
                value,step = (float(tensor),e.step)
                values.append(value)
                steps.append(step)
    return values,steps

def smooth(x,window_len=11,window='hanning'):
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=np.ones(window_len,'d')
    else:  
            w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]
    

scalar_files = glob.glob("../../master_thesis_ai/tb_logs_extension/trec-covid_*/version_0/events*")
teacher_files = glob.glob("../../master_thesis_ai/tb_logs_extension/trec-covid_*/version_0/Means of teachers_mean_cross-encoder/ms-marco-MiniLM-L-6-v2/*")

fig = plt.figure()
for file in scalar_files:
    val,st = get_scalar_run_tensorboard("Distill_loss",
                                file)
    # Search for the pattern in the input string
    match = re.search(pattern, file)

    # Extract the number if a match is found
    if match:
        extracted_number = match.group(0)
        plt.plot(st,smooth(np.array(val), window_len = 100), label = f"Remine every {extracted_number}", alpha = 0.6)   
plt.legend()
plt.savefig("fig_distill.png")

fig = plt.figure()
print(teacher_files)
for file in teacher_files:
    val,st = get_scalar_run_tensorboard("Means of teachers",
                                file)
    # Search for the pattern in the input string
    match = re.search(pattern, file)

    # Extract the number if a match is found
    if match:
        extracted_number = match.group(0)
        plt.plot(st,smooth(np.array(val), window_len = 10000), label = f"Mean margin for {extracted_number}", alpha = 0.6)   
plt.legend()
plt.savefig("fig_margin.png")
