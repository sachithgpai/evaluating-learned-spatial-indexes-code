# %%
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from joblib import Parallel, delayed
import os,sys
import shutil

#%%
PROJECT_ROOT = Path(__file__).parent
experiment_folder = PROJECT_ROOT/sys.argv[1]
if experiment_folder.exists():
    shutil.rmtree(experiment_folder)
experiment_folder.mkdir(parents=True,exist_ok=True)


a=0.0001
b=0.003
ScalerMatrix= (b-a)*np.ones((2,2))+a

num_samples_per_group = 5
num_different_datasets = 5
N = int(sys.argv[2]) #1000000                                             # Number of data points
M = int(sys.argv[3])                                                 # Number of queries
selectivities_arr = [ "00064", "00256", "01024", "04096", "16384"]

rng = np.random.default_rng()
scales = np.flip(np.logspace(0.1, 1.4, num=num_samples_per_group))  # the ratios to scale the data points from the initial 
query_scales = np.flip(np.logspace(0.1, 1.4, num=num_samples_per_group))


print(scales)
print(query_scales)


#%%

"""
Function to sample iteratively from Gaussian.
"""
def sampleFromGaussian(mean:np.ndarray, cov:np.ndarray,count:int):

    X = rng.random((int(count*0.0001),2))

    while(X.shape[0]<count):
        sample = rng.multivariate_normal(mean, cov, max(1000,count))
        sample = sample[sample[:,0]>0]
        sample = sample[sample[:,1]>0]
        sample = sample[sample[:,0]<1]
        sample = sample[sample[:,1]<1]
        X = np.vstack((X,sample))
    
    return X    



# %%


def generateThreeTypesOfQueries(data_dist,cur_dataset_path,sample_num):
    Q_dist_arr =[]
    uniform_Query_E = entropy(rng.random((M,2)),nbins=32)

    # Sampling a different set of skewed clusters to act as query_dist
    num_query_clusters = 5 #np.random.randint(3,9)
    num_query_centers_per_cluster = (M//num_query_clusters)+1


    mean_arr = rng.choice(data_dist,num_query_clusters,replace=False)                 # holds 2d points which are cluster centers
    
    cov_arr = []
    for cluster in range(num_query_clusters):
        A = np.random.rand(2, 2)*2-1
        cov_arr.append(np.multiply(np.dot(A, A.transpose()),ScalerMatrix))


    entropy_list = []
    for query_sample_num in range(1,num_samples_per_group+1):
        Q = None
        scale = query_scales[query_sample_num-1]
        for cluster in range(num_query_clusters):
            cov = cov_arr[cluster].copy()
            cov[0,0]*=scale
            cov[1,1]*=scale
            if Q is None:
                Q = sampleFromGaussian(mean=mean_arr[cluster],cov=cov,count=num_query_centers_per_cluster)
            else:
                Q = np.vstack((Q,sampleFromGaussian(mean=mean_arr[cluster],cov=cov,count=num_query_centers_per_cluster)))

        rng.shuffle(Q,axis=0)
        Q = Q[:M]
        Q_dist_arr.append(Q)
        entropy_list.append([query_sample_num,entropy(Q,nbins=32)/uniform_Query_E])


    f = open(cur_dataset_path/'queries'/'entropy_values','a')
    for ent in entropy_list:
        f.write('{} {} {}\n'.format(sample_num,ent[0],ent[1]))
    f.close()

    for selec in selectivities_arr:
        parallelSampling(selec,data_dist,Q_dist_arr,cur_dataset_path,sample_num)


def parallelSampling(selec,data_dist,Q_dist_arr,cur_dataset_path,sample_num):
    w = math.sqrt((int(selec)/10000.0)/400.0)
    
    ## Follows Uniform dist
    (cur_dataset_path/'queries'/'uniformDist').mkdir(parents=True, exist_ok=True)
    uniform_query_dist = rng.random((M,2))# to provide same query dist to area and count based.
    sampleAreaBasedClusteredQueries(uniform_query_dist,w,cur_dataset_path/'queries'/'uniformDist'/'{}_{}_areabased'.format(sample_num,selec))

    ## Follows Data dist
    (cur_dataset_path/'queries'/'dataDist').mkdir(parents=True, exist_ok=True)
    sampleAreaBasedClusteredQueries(data_dist[:M],w,cur_dataset_path/'queries'/'dataDist'/'{}_{}_areabased'.format(sample_num,selec))

    ## Follows other dist
    (cur_dataset_path/'queries'/'otherDist').mkdir(parents=True, exist_ok=True)
    for query_sample_num,query_dist in enumerate(Q_dist_arr):
        np.savetxt(cur_dataset_path/'queries'/'otherDist'/'{}_{}_querycenteres_{}'.format(sample_num,selec,query_sample_num+1),query_dist,fmt='%.9f')
        sampleAreaBasedClusteredQueries(query_dist,w,cur_dataset_path/'queries'/'otherDist'/'{}_{}_areabased_{}'.format(sample_num,selec,query_sample_num+1))

    return True

def sampleAreaBasedClusteredQueries(query_dist,w,filename):
    allQueries = []
    x,y = query_dist.T
    allQueries = np.stack([np.clip(x-w,0,1),
                        np.clip(y-w,0,1),
                        np.clip(x+w,0,1),
                        np.clip(y+w,0,1)],axis=-1)
    np.savetxt(filename, allQueries, delimiter=' ',fmt='%.9f')




# Function to calculate the skewness of data.
def entropy(X,nbins=512):
    hist_prob = np.ravel(np.histogram2d(X[:,0],X[:,1], bins = nbins)[0]/X.shape[0])
    hist_prob = np.where(hist_prob < 1e-9, 1e-9, hist_prob)
    entropy = -np.sum(np.multiply(hist_prob, np.log2(hist_prob)))
    return entropy





#%%



def generateDataset():
    X = rng.random((N,2))
    uniform_E = entropy(X)

    for dataset_num in range(1,num_different_datasets+1):
        cur_dataset_path = experiment_folder/str(dataset_num)
        (cur_dataset_path/'datapoints/plots').mkdir(parents=True, exist_ok=True)
        (cur_dataset_path/'queries').mkdir(parents=True, exist_ok=True)


        num_clusters = 10 #np.random.randint(10,20)
        num_points_per_cluster = (N//num_clusters)+1
        mean_arr = 0.9*rng.random((num_clusters,2))+0.05       # holds 2d points which are cluster centers
        cov_arr = []                                # holds the covariance matrices of guassian clusters.

        for cluster in range(num_clusters):
            A = np.random.rand(2, 2)*2-1
            cov_arr.append(np.multiply(np.dot(A, A.transpose()),ScalerMatrix))

        entropy_list = []
        for sample_num in range(1,num_samples_per_group+1):
            X = rng.random((10000,2))
            # X = np.vstack((X,np.array([[0,0],[0,1],[1,0],[1,1]])))
            scale = scales[sample_num-1]
            for cluster in range(num_clusters):
                cov =cov_arr[cluster].copy()
                cov[0,0]*=scale
                cov[1,1]*=scale
                X = np.vstack((X,sampleFromGaussian(mean=mean_arr[cluster],cov=cov,count=num_points_per_cluster)))

            rng.shuffle(X,axis=0)
            X = X[:N]
            E = entropy(X)
            entropy_list.append([sample_num,E*1.0/uniform_E])


            generateThreeTypesOfQueries(X, cur_dataset_path, sample_num)
            np.savetxt(cur_dataset_path/'datapoints'/'{}'.format(sample_num),X,fmt='%.9f')
            
            plt.scatter(X[:100000,0],X[:100000,1],alpha=0.2,s=0.2)
            plt.savefig(cur_dataset_path/'datapoints'/'plots'/'{}'.format(sample_num),dpi=400)
            plt.close()
            

        f = open(cur_dataset_path/'datapoints'/'entropy_values','a')
        for ent in entropy_list:
            f.write('{} {}\n'.format(ent[0],ent[1]))
        f.close()

        


#%%

print(experiment_folder)


#%%

generateDataset()

    
# %%


