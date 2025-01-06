import pandas as pd
import numpy as np



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os 


#Read dataset 
data_path=os.path.join(os.getcwd(),'data','Branch_data.csv')
#print('data_path: ',data_path)
df=pd.read_csv(data_path)


print(df.head())

# Dataset

for i in range(1,101):
    num_class=i
    branches = df.loc[0:num_class,'Branches'].to_numpy()
    gflops=df.loc[:num_class,'GFLOPS'].to_numpy()

    # Normalization
    scaler = MinMaxScaler(feature_range=(0,1))
    flexibility_norm = 1-(branches/num_class)#scaler.fit_transform(branches.reshape(1,-1)).flatten()
    gflops_norm = scaler.fit_transform(gflops.reshape(-1, 1)).flatten()

    #print(gflops_norm)

    #print(flexibility_norm)
    # Objective function
    w1= 0.02
    w2=1-w1
    #scores = w1*flexibility_norm + w2*gflops_norm
    scores=np.sqrt((w1*flexibility_norm-w2*gflops_norm)**2)


    # Find optimal branch count
    optimal_index = np.argmin(scores)
    optimal_branches = branches[optimal_index]
    print('num class:',num_class,'branches: ',optimal_branches)

'''# Visualization
plt.figure(figsize=(12, 6))
plt.plot(branches, w1*flexibility_norm, label='Flexibility (normalized)', marker='o', markersize=4)
plt.plot(branches, w2*gflops_norm, label='FLOPS (normalized)', marker='o', markersize=4)
plt.axvline(optimal_branches, color='r', linestyle='--', label=f'Optimal Branches: {optimal_branches}')
plt.title('Trade-off Between Flexibility and FLOPS')
plt.xlabel('Number of Branches')
plt.ylabel('Normalized Values')
plt.legend()
plt.grid(True)
plt.show()'''



