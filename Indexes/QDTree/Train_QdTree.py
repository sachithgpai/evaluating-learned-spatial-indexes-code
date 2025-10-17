#%%

import numpy as np
from collections import defaultdict
import random
from collections import deque
from IPython import get_ipython
if get_ipython() is not None and __name__ == "__main__":
    notebook = True
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
else:
    notebook = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from tensordict import TensorDict , merge_tensordicts, dense_stack_tds
from tensordict.nn import TensorDictModule, InteractionType
from torch import nn

from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ProbabilisticActor, ValueOperator,ActorValueOperator, MaskedOneHotCategorical, ReparamGradientStrategy
from torchrl.objectives import ClipPPOLoss, PPOLoss
from torchrl.objectives.value import GAE, TD0Estimator
from tqdm import tqdm
import pprint
from timeit import default_timer as timer
from pathlib import Path
from QdTree_environment import QdNode, QdTreeEnvironment, ActionSpaceSize
import shutil
from collections import Counter
from typing import Optional
import sys, os, time
PROJECT_ROOT = Path('/scratch/project_2005865/sachithp/experiments-md-index')
DATASET_FOLDER_NAME = None
SAMPLE_RATIO = 0.1



rng = np.random.default_rng(42)


#%%
"""
A function to train and save a QD-tree index based given parameters.
    dataset_name            :   Name of the dataset (eg. Dataset-16M)
    datapoint_sample_num    :   The sample dataset num
    datapoint_entropy_id    :   The entropy id within dataset num
    query_type              :   query_type
    selectivity             :   selectivity
    query_entropy_id        :   query_entropy 
    page_size               :   Page size of the constructed index
"""
def trainQDTreeAndSaveBestTree( datapoint_sample_num, datapoint_entropy_id, query_type, selectivity, query_entropy_id, page_size, area_or_count_based):


    print('trainQDTreeAndSaveBestTree', datapoint_sample_num, datapoint_entropy_id, query_type, selectivity, query_entropy_id, page_size)
    device = "cpu"                      # if not torch.has_cuda else "cuda:0"
    num_cells = 512                     # number of cells in each layer i.e. output dim.
    lr = 1e-5                           # learning rate
    max_grad_norm = 0.5                 # gradient normalization that should help in PPO
    num_of_episodes = 50               # Number of iterations of optimization to run:
    num_epochs = 2                      # optimisation steps per iteration of data collected
    mini_batch_size = 16                    
    gamma = 0.97                        # discounted expected reward
    lmbda = 0.93                        # trajectory discount
    # ClipPPOLoss params
    clip_epsilon = 0.2 # clip value for PPO loss: see the equation in the intro for more context.
    entropy_coef=0.01 # default 0.01
    critic_coef=0.01 # default 1
    loss_critic_type="l2"
    normalize_advantage=False
    separate_losses=False
    grad_method=ReparamGradientStrategy.PassThrough
    torch.manual_seed(seed=123)


    ######### The Neural Networks for Agents #########
    module_hidden = nn.Sequential(
        nn.Linear(ActionSpaceSize,num_cells, device=device),
        nn.PReLU(),
        nn.Linear(num_cells,num_cells, device=device),
    )

    tensordict_module_hidden = TensorDictModule(
        module_hidden, in_keys=["observation"], out_keys=["hidden"]
    )

    action_layer = nn.Sequential(
        nn.Linear(num_cells,num_cells, device=device),
        nn.PReLU(),
        nn.Linear(num_cells, ActionSpaceSize, device=device),
    )
    module_action_layer = TensorDictModule( action_layer, in_keys=["hidden"], out_keys=["logits"], )

    td_module_action = ProbabilisticActor(
        module=module_action_layer,
        in_keys=["logits","mask"],
        out_keys=["action"],
    #    distribution_class=CustomMaskedOneHotCategorical,
        distribution_class=MaskedOneHotCategorical,

        return_log_prob=True, # we'll need the log-prob for the numerator of the importance weights
        default_interaction_type=InteractionType.RANDOM,
        distribution_kwargs=dict(grad_method=grad_method)
    )

    # module_value_layer = nn.Linear(num_cells, 1, device=device)
    module_value_layer = nn.Sequential(
        nn.Linear(num_cells,num_cells, device=device),
        nn.PReLU(),
        nn.Linear(num_cells, 1, device=device),
    )
    td_module_value = ValueOperator(module=module_value_layer,in_keys=["hidden"],)


    td_module = ActorValueOperator(tensordict_module_hidden, td_module_action, td_module_value)
    value_module = td_module.get_value_operator()
    policy_module = td_module.get_policy_operator()





    
    ####################### MAIN ##########################
    datapoints = np.loadtxt(PROJECT_ROOT/'Datasets'/DATASET_FOLDER_NAME/datapoint_sample_num/'datapoints'/datapoint_entropy_id,delimiter=' ')
    queries = np.loadtxt(PROJECT_ROOT/'Datasets'/DATASET_FOLDER_NAME/datapoint_sample_num/'queries'/'otherDist'/f'{datapoint_entropy_id}_{selectivity}_{area_or_count_based}_{query_entropy_id}',delimiter=' ')


    index_save_folder = PROJECT_ROOT/'Experiments'/DATASET_FOLDER_NAME/'TrainedIndexes'/'QDTree'
    index_save_folder.mkdir(parents=True,exist_ok=True)
    tree_name = f"P_{page_size}_D_{datapoint_sample_num}_DE_{datapoint_entropy_id}_Q_{query_entropy_id}_S_{selectivity}_{area_or_count_based}"

    rng.shuffle(datapoints)
    rng.shuffle(queries)
    print(f"Tree: P_{page_size}_D_{datapoint_sample_num}_DE_{datapoint_entropy_id}_Q_{query_entropy_id}_S_{selectivity}. Sample_ratio:",SAMPLE_RATIO," Page-Size:",int(page_size*SAMPLE_RATIO*64))


    # initializing the environment    
    obj = QdTreeEnvironment(datapoints[:int(datapoints.shape[0]*SAMPLE_RATIO)],queries,int(page_size*SAMPLE_RATIO*64))
    tensordict_data,_,_,_,_ = obj.trainQDTreeCreateEpisodicStateActionReward(policy_module)

    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(tensordict_data.size()[0]*2),
        sampler=SamplerWithoutReplacement(),
    )


    
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus= True,#bool(entropy_eps),
        entropy_coef=entropy_coef,
        # these keys match by default but we set this for completeness
        # value_target_key=advantage_module.value_target_key,
        critic_coef=critic_coef,
        loss_critic_type=loss_critic_type,
        normalize_advantage=normalize_advantage,
        separate_losses=separate_losses
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,0.999)

    train_start_time = time.process_time()
    best_reward = -np.inf
    best_root = None
    print('########## Starting Episodes #############')
    for i in range(num_of_episodes):
        print('Episode {}/{}'.format(i,num_of_episodes))
        # we now have a batch of data to work with. Let's learn something from it.
        tensordict_data, avg_root_reward, average_node_depth, _best_reward, _best_root = obj.trainQDTreeCreateEpisodicStateActionReward(policy_module)
        loss_array = []
        loss_objective_array, loss_critic_array, loss_entropy_array, policy_entropy_array = [],[],[],[]
        
        if _best_reward > best_reward:
            best_reward = _best_reward
            best_root=_best_root
            obj.root = best_root
            obj.saveQdTree(str(index_save_folder/f"{tree_name}.txt"))

        for j in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            with torch.no_grad():
                advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.empty()
            replay_buffer.extend(data_view.cpu())

            for k in range(int(np.ceil(tensordict_data.size()[0] / mini_batch_size))):
                subdata = replay_buffer.sample(mini_batch_size).to(device)
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )


                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                loss_array.append(loss_value.item())
                loss_objective_array.append(loss_vals["loss_objective"].item())
                loss_critic_array.append(loss_vals["loss_critic"].item())
                loss_entropy_array.append(loss_vals["loss_entropy"].item())
                policy_entropy_array.append(loss_vals["entropy"].item())

                # this is not strictly mandatory but it's good practice to keep your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        scheduler.step()

    print(time.process_time() - train_start_time)  
    with open(str(index_save_folder/f"{tree_name}.time"), 'w') as fileoperator:
        fileoperator.write('{}\n'.format(time.process_time() - train_start_time))


# %%
page_size_array = [32, 64, 128, 256, 512, 1024, 2048, 4096]
selectivities_arr = [ "00064", "00256", "01024", "04096", "16384"]


if __name__ == '__main__':
    DATASET_FOLDER_NAME = str(sys.argv[1])
    data_sample_num = int(sys.argv[2])
    data_ent_id = int(sys.argv[3])
    page_size = int(sys.argv[4])
    query_ent_id = int(sys.argv[5])
    selectivity  =  selectivities_arr[int(sys.argv[6])]
    
    trainQDTreeAndSaveBestTree(str(data_sample_num),str(data_ent_id),'otherDist',selectivity,str(query_ent_id),page_size,'areabased')

    trainQDTreeAndSaveBestTree(str(data_sample_num),str(data_ent_id),'otherDist',selectivity,str(query_ent_id),page_size,'countbased')
