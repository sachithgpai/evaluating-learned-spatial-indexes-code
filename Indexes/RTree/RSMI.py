#%%
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys, os, time
from pathlib import Path
import zCurve 
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from scipy.stats import rankdata
from time import time

RSMI_SPLIT_CNT = 4                 # creates a branching factor of 16
SAMPLING_RATIO = 0.1
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DATASET_FOLDER_NAME = ''
RTreeStructure = []
RTreeStructure_LR = []

#%%

class RSMINode(torch.nn.Module):
    def __init__(self,num_classes):
        super(RSMINode, self).__init__()
        self.linear1 = torch.nn.Linear(2, 64)
        self.linear2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        # output = self.linear2(F.leaky_relu(self.linear1(x))) 
        output = F.log_softmax(self.linear2(F.leaky_relu(self.linear1(x)))) # NLLTest
        return output


class TargetDatasetForRSMI(Dataset):
    def __init__(self, inp, out):
        self.X  = inp
        self.y = out

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]



def TrainRSMINode(datapoints,index_save_folder,page_size,tree_write_file,node_name_str='0'):
    global RTreeStructure
    num_splits_per_dim = min(RSMI_SPLIT_CNT,np.ceil(np.sqrt(datapoints.shape[0]/(page_size*4))).astype(int))
    num_boxes = num_splits_per_dim**2

    minx,miny,maxx,maxy = *(np.min(datapoints[:,:2],axis=0)),*(np.max(datapoints[:,:2],axis=0))

    # print("|N|:",datapoints.shape[0],' numsplits:',num_splits_per_dim,'min-max :',minx,miny,maxx,maxy)
    if datapoints.shape[0]<=page_size:
        RTreeStructure.append([0,0,node_name_str]+list(np.min(datapoints[:,:2],axis=0))+list(np.max(datapoints[:,:2],axis=0)))
        tree_write_file.write('{} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(0,minx,miny,maxx,maxy))
        tree_write_file.write('{}\n'.format(datapoints.shape[0]))
        for datapoint_ in datapoints:
            tree_write_file.write('{:.9f} {:.9f}\n'.format(datapoint_[0],datapoint_[1]))
        return

    ## Neural network training at the last layer is error-prone. At final level make STR-like splits
    if datapoints.shape[0]<=page_size*16:
        num_splits_per_dim = min(RSMI_SPLIT_CNT,np.ceil(np.sqrt(datapoints.shape[0]/(page_size))).astype(int))
        num_boxes = num_splits_per_dim**2
        prepared_datapoints = PrepareDataForTraining(datapoints,num_splits_per_dim)

        tree_write_file.write('{} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(num_boxes,minx,miny,maxx,maxy))
        # print(prepared_datapoints[:,2])
        # input()


        for child_num in range(num_boxes)[::-1]:
            child_datapoints = prepared_datapoints[prepared_datapoints[:,2]==child_num]
            # print('child_num',child_num)

            if child_datapoints.shape[0]>0:
                TrainRSMINode(child_datapoints,index_save_folder,page_size,tree_write_file,node_name_str+'_{}'.format(child_num))


        return

    

    sampled_datapoints = datapoints[np.random.choice(datapoints.shape[0],replace=False,size=min(datapoints.shape[0],4096))].copy()

    sampled_datapoints =  PrepareDataForTraining(sampled_datapoints,num_splits_per_dim)


    # np.savetxt(index_save_folder/(node_name_str+'_TARGET_bincounts.txt'),np.bincount(sampled_datapoints[:,2].astype(int),minlength=num_boxes),fmt='%d') #DEBUGFILES
    if False:  #DEBUGFILES
        fig,ax = plt.subplots(figsize=(10,10))
        ax.scatter(sampled_datapoints[:,0],sampled_datapoints[:,1],c=sampled_datapoints[:,2],s=1410/np.sqrt(sampled_datapoints.shape[0]))
        fig.savefig(index_save_folder/(node_name_str+'_RSMI_points_Target.png'))
        plt.close()
    # print('sampled_datapoints',sampled_datapoints.shape)


    inp = torch.from_numpy(sampled_datapoints[:,0:2].astype(np.float32))
    # out = torch.from_numpy(np.eye(num_boxes,dtype=np.float32)[sampled_datapoints[:,2].astype(np.int32)])
    out = torch.from_numpy(sampled_datapoints[:,2].astype(np.float32)).type(torch.LongTensor)  # NLLTest


    number_of_epochs =256         
    net = RSMINode(num_boxes)

    criterion = torch.nn.CrossEntropyLoss() 
    # criterion = torch.nn.NLLLoss()        # NLLTest
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.8, total_iters=number_of_epochs)


    hist = []
    net.train()

    for epoch in range(number_of_epochs):  # loop over the dataset;

        pred = net(inp)
        loss = criterion(pred,out)

        # Backpropagation
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        hist.append(float(criterion(net(inp),out)))
        optimizer.zero_grad()

    net.eval()


    # traced = torch.jit.trace(net, inp) #MAKEITRTREE
    # traced.save(index_save_folder/(node_name_str+'.pt'))

    if False:
        fig,ax = plt.subplots(figsize=(10,10))
        box_sampled = torch.argmax(net(inp), dim=1).int().reshape(-1)
        ax.scatter(sampled_datapoints[:,0],sampled_datapoints[:,1],c=box_sampled.detach().numpy(),s=1410/np.sqrt(sampled_datapoints.shape[0]))
        fig.savefig(index_save_folder/(node_name_str+'_RSMI_points_Learned.png'))
        plt.close()


    #split data points into groups 
    pred = net(torch.from_numpy(datapoints[:,0:2].astype(np.float32)))
    box = torch.argmax(pred, dim=1).int().reshape(-1)
    actual_bincounts = torch.bincount(box,minlength=num_boxes).numpy()

    # np.savetxt(index_save_folder/(node_name_str+'_ACTUAL_bincounts.txt'),actual_bincounts,fmt='%d')
    del net




    # plt.plot(range(number_of_epochs),hist)
    # plt.title('|N|:{} |Branch|:{}'.format(datapoints.shape[0],num_boxes))
    # plt.tight_layout()
    # plt.yscale('log')
    # plt.savefig(index_save_folder/(node_name_str+'_learning_rate.png'))
    # plt.close()

    # print(' {} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(np.count_nonzero(actual_bincounts),*(np.min(datapoints[:,:2],axis=0)),*(np.max(datapoints[:,:2],axis=0))),actual_bincounts)
    
    tree_write_file.write('{} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(np.count_nonzero(actual_bincounts),minx,miny,maxx,maxy))


    number_of_children = 0
    for child_num in range(num_boxes)[::-1]:
        child_datapoints = datapoints[box==child_num]
        if child_datapoints.shape[0]>0:
            TrainRSMINode(child_datapoints,index_save_folder,page_size,tree_write_file,node_name_str+'_{}'.format(child_num))
            number_of_children += 1
    
    if(number_of_children != np.count_nonzero(actual_bincounts)):
        print('The number of children mismatched at {}',node_name_str)
    # RTreeStructure.append([number_of_children,num_splits_per_dim**2,node_name_str]+list(np.min(datapoints[:,:2],axis=0))+list(np.max(datapoints[:,:2],axis=0))) #MAKEITRTREE




def PrepareDataForTraining(datapoints,num_splits_per_dim):    
    datapoints = datapoints[np.argsort(datapoints[:,0])]
    splits_per_branch_x = datapoints.shape[0]//num_splits_per_dim
    splits_per_branch_x_remainder = datapoints.shape[0]%num_splits_per_dim

    start_x = 0
    end_x = 0
    for x_ix in range(num_splits_per_dim):
        start_x = end_x
        end_x += splits_per_branch_x

        if splits_per_branch_x_remainder != 0:
            end_x+=1
            splits_per_branch_x_remainder-=1


        datapoints[start_x:end_x] = datapoints[start_x:end_x][np.argsort(datapoints[start_x:end_x,1])]


        splits_per_branch_y = (end_x-start_x)//num_splits_per_dim
        splits_per_branch_y_remainder = (end_x-start_x)%num_splits_per_dim       
        start_y = 0
        end_y = start_x
        
        for y_ix in range(num_splits_per_dim):
            start_y = end_y
            end_y += splits_per_branch_y

            if splits_per_branch_y_remainder != 0:
                end_y+=1
                splits_per_branch_y_remainder-=1
            
            datapoints[start_y:end_y,2] = zCurve.interlace(y_ix,x_ix)
    
    # To make the function to learn more uniform. Otherwise there are missing values in the z-ordered values and it causes imbalance in training out boxes.
    datapoints[:,2] = rankdata(datapoints[:,2], method='dense') - 1

    # datapoints[:,2] /= np.max(datapoints[:,2])
    return datapoints



    



# %%
page_size_array = [32, 64, 128, 256, 512, 1024, 2048, 4096]
selectivities_arr = [ "00064", "00256", "01024", "04096", "16384"]


if __name__ == '__main__':
    DATASET_FOLDER_NAME = str(sys.argv[1])
    data_sample_num = int(sys.argv[2])
    data_ent_id = int(sys.argv[3])
    page_size = int(sys.argv[4])

    index_save_folder_name = f"P_{page_size}_D_{data_sample_num}_DE_{data_ent_id}"
    index_save_folder = PROJECT_ROOT/'Experiments'/DATASET_FOLDER_NAME/'TrainedIndexes'/'RSMI'/index_save_folder_name
    # index_save_folder.mkdir(parents=True,exist_ok=True) # DEBUGFILES


    datapoints = np.loadtxt(PROJECT_ROOT/'Datasets'/DATASET_FOLDER_NAME/str(data_sample_num)/'datapoints'/str(data_ent_id),delimiter=' ')
    np.random.shuffle(datapoints)

    datapoints = np.hstack((datapoints,np.zeros((datapoints.shape[0],1)))) # block_id
    
    print(' datapoints: {}/{}/datapoints/{}'.format(DATASET_FOLDER_NAME,data_sample_num,data_ent_id))



    tree_write_file = open(PROJECT_ROOT/'Experiments'/DATASET_FOLDER_NAME/'TrainedIndexes'/'RSMI'/(index_save_folder_name+'.tree'),'w')

    t1 = time()
    TrainRSMINode(datapoints,index_save_folder,page_size,tree_write_file)
    t2 = time()

    tree_write_file.close()

    with open(index_save_folder.parent/(index_save_folder_name+'.time'),'w') as fopen:
        fopen.write('%.6f'%(t2-t1))



    # with open(index_save_folder/'tree.txt',"a") as fopen:  # MAKEITRTREE
    #     for num_children, num_splits_per_dim, node_name_str, lowx, lowy, highx, highy in reversed(RTreeStructure):
    #         fopen.write("{} {} {} {} {} {} {} {}\n".format(num_children, num_splits_per_dim, node_name_str, node_name_str.split('_')[-1], lowx, lowy, highx, highy))

    # RTreeStructure.clear()





#%%


    # t1 = time()
    # TrainRSLRINode(datapoints,index_save_folder,page_size)
    # t2 = time()

    # with open(index_save_folder.parent/(index_save_folder_name+'_LR.time'),'a') as fopen:
    #     fopen.write('%.6f'%(t2-t1))

    # with open(index_save_folder/'tree_LR.txt',"a") as fopen:
    #     for num_children, num_splits_per_dim, node_name_str, lowx, lowy, highx, highy in reversed(RTreeStructure_LR):
    #         fopen.write("{} {} {} {} {} {} {} {}\n".format(num_children, num_splits_per_dim, node_name_str,node_name_str.split('_')[-1], lowx, lowy, highx, highy))

    # RTreeStructure_LR.clear()


# def TrainRSLRINode(datapoints,index_save_folder,page_size,node_name_str='0'):
#     global RTreeStructure_LR

#     num_splits_per_dim = RSMI_SPLIT_CNT #min(RSMI_SPLIT_CNT,np.floor(np.sqrt(datapoints.shape[0]/(page_size*10))).astype(int))

#     if datapoints.shape[0]<10000:
#         RTreeStructure_LR.append([0,0,node_name_str]+list(np.min(datapoints[:,:2],axis=0))+list(np.max(datapoints[:,:2],axis=0)))
#         return

#     datapoints =  PrepareDataForTraining(datapoints,num_splits_per_dim)


#     if True:
#         fig,ax = plt.subplots(figsize=(10,10))
#         ax.scatter(datapoints[:,0],datapoints[:,1],c=datapoints[:,2],s=141.0/np.sqrt(datapoints.shape[0]))
#         fig.savefig(index_save_folder/(node_name_str+'_LR_points_Target.png'))
#         plt.close()

#     inp = datapoints[:,0:2].astype(np.float32)
#     inp = np.hstack((inp,np.square(inp),np.multiply(inp[:,0],inp[:,1]).reshape(-1,1)))

#     out = datapoints[:,2].astype(np.float32).reshape(-1,1)
    

#     lr_model = LinearRegression(positive=True).fit(inp,out)


#     with open(index_save_folder/(node_name_str+'_LR.txt'),"a") as fopen:
#         fopen.write("{} {} {} {} {}".format(lr_model.coef_[0,0],lr_model.coef_[0,1],lr_model.coef_[0,2],lr_model.coef_[0,3],lr_model.coef_[0,4],lr_model.intercept_[0]))

#     #split data points into groups TODO:
#     pred = lr_model.predict(inp)
#     num_boxes = num_splits_per_dim**2
#     box = np.clip(np.round(pred*(num_boxes-1)),a_min=0,a_max=num_boxes-1).astype(int).flatten()
#     np.savetxt(index_save_folder/(node_name_str+'_bincounts_LR.txt'),np.bincount(box.flatten()),fmt='%d')

#     if True:
#         fig,ax = plt.subplots(figsize=(10,10))
#         ax.scatter(datapoints[:,0],datapoints[:,1],c=(box/(num_boxes-1)),s=141.0/np.sqrt(datapoints.shape[0]))
#         fig.savefig(index_save_folder/(node_name_str+'_LR_points_Learned.png'))
#         plt.close()


#     number_of_children = 0
#     for child_num in range(num_boxes)[::-1]:
#         child_datapoints = datapoints[box==child_num]
#         if child_datapoints.shape[0]>0:
#             TrainRSLRINode(child_datapoints,index_save_folder,page_size,node_name_str+'_{}'.format(child_num))
#             number_of_children += 1
    
#     RTreeStructure_LR.append([number_of_children,num_splits_per_dim**2,node_name_str]+list(np.min(datapoints[:,:2],axis=0))+list(np.max(datapoints[:,:2],axis=0)))







# # %%
    ## Doing some debugging
    # if True:


    #     plt.plot(range(number_of_epochs),hist)
    #     plt.yscale('log')
    #     plt.title('|NN-MSE|:{:.4f} |LR-MSE|:{:.4f}'.format(root_mean_squared_error(lr_model.predict(inp.numpy()),out.numpy()),root_mean_squared_error(pred.detach().numpy(),out.numpy())))
    #     plt.savefig(index_save_folder/(node_name_str+'_learning_rate.png'))
    #     plt.close()


    #     plt.hist( box,bins=num_boxes)
    #     plt.title('|N|:{} |Branching|:{}'.format(datapoints.shape[0],num_splits_per_dim))
    #     plt.savefig(index_save_folder/(node_name_str+'_split_hist.png'))
    #     plt.close()


    #     sk_pred = lr_model.predict(inp.numpy())
    #     sk_box = clip(np.round(sk_pred*(num_boxes-1)),a_min=0,a_max=num_boxes-1).astype(int)clip(np.round(sk_pred*(num_boxes-1)),a_min=0,a_max=num_boxes-1).astype(int)
    #     plt.hist(sk_box,bins=num_boxes)
    #     plt.title('|N|:{} |Branching|:{}'.format(datapoints.shape[0],num_splits_per_dim))
    #     plt.savefig(index_save_folder/(node_name_str+'_LINEAR_REGRESSION_split_hist.png'))
    #     plt.close()


        
    #     out_box = np.clip(np.round(out.numpy()*(num_boxes-1)),a_min=0,a_max=num_boxes-1).astype(int)
    #     plt.hist(out_box,bins=num_boxes)
    #     plt.title('|N|:{} |Branching|:{}'.format(datapoints.shape[0],num_splits_per_dim))
    #     plt.savefig(index_save_folder/(node_name_str+'_TARGET_split_hist.png'))
    #     plt.close()


    #     np.savetxt(index_save_folder/(node_name_str+'_bincounts.txt'),torch.bincount(box).numpy(),fmt='%d')

    #     np.savetxt(index_save_folder/(node_name_str+'_TARGET_bincounts.txt'),np.bincount(out_box.flatten())np.bincount(out_box.flatten()),fmt='%d')

    #     plt.plot(np.arange(datapoints.shape[0]),np.sort(datapoints[:,2]))
    #     plt.savefig(index_save_folder/(node_name_str+'_RAWTARGET_func.png'))
    #     plt.close()
