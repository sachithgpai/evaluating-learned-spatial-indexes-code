
""" Q-d tree environment. 

This code emulates the Q-d tree environement.
"""



import numpy as np
from collections import deque
import torch
from tensordict.nn import TensorDictModule, InteractionType
from tensordict import TensorDict , merge_tensordicts, dense_stack_tds

from timeit import default_timer as timer
from qd_tree_query_cost_calculator import node_query_overlap_count
from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy

ActionSpaceSize = 1024              # Size of the action space
precomputed_gamma = 0.9           # gamma to use when we are manually computing the value targe recursively.
num_traectories_per_iter = 1     # How many different trajectories to include in an iteration.
page_frac = 0.5                     # how small can a leaf be 


class QdNode:
    def __init__(self,stateRepre,points):
        self.isLeaf = True
        self.predicate = None           # the id of the predicate that was used at this node


        """
        Each state is represented by a set of booleans to convey the set of valid predicates.
        Consider a grid structure imposed on the dataspace based on a set of predicates split points. Each valid node can represented by the set of predicates which can split the current node further. At the beggining all predicates are valid therefore they are all acceptable.
        Here the valid predicates for two dims are represented by a [2,ActionSpaceSize//2] vector.
        """
        self.stateRepresentation = stateRepre.copy()    # the predicate vector for this node.
        self.mask=None
        self.pointsArray = points.copy()
        self.mbr = None                                 # lx, ly, hx, hy



        self.leftChild = None
        self.rightChild = None

        self.numberOfPoints = points.shape[0]
        self.queryOverlapCount = None
        
        
        self.S = None
        self.reward = None
        self.oneLevelReward = None
        self.precomputedValueTarget = None

        self.policy_tuple = None


        

class QdTreeEnvironment:

    def __init__(self, points, queries,page_size=4096):
        """
        points : np.array((N,2))
            A list of points. <double,double>
        predicates : np.array((2*P,2)) <int,double>
            A list of candidate cuts sampled as the lowerbound of the queries. Each predicate indicates (x/y </>= split values).
            For ease of use the array has the X(first P) and Y(last P) predicates sorted according to their values.
        """
        self.root = None
        self.explorationQueue = deque()
        self.pointsList = points
        self.currentNode = None
        self.queries = queries.copy()
        self.predicatesList = []
        self.REEVALUATES = 0
        self.PageSize = page_size                    # The leaf size up until which the program should try to split.

        # # Extracting evenly spaced predicates.        
        # for a in np.linspace(0.0, 1.0 , num=(ActionSpaceSize//2)+2)[1:-1]:
        #     self.predicatesList.append((0,a))
        #     self.predicatesList.append((1,a))
        # self.predicatesList.sort()


        
        # Extracting data spaced predicates
        points = points[points[:, 0].argsort()]
        for a in np.linspace(0.0, points.shape[0]-1 , num=(ActionSpaceSize//2)+2,dtype=int)[1:-1]:
            self.predicatesList.append((0,points[a,0]))

        points = points[points[:, 1].argsort()]
        for a in np.linspace(0.0, points.shape[0]-1 , num=(ActionSpaceSize//2)+2,dtype=int)[1:-1]:
            self.predicatesList.append((1,points[a,1]))



    def recursivelySplitAndCalculateRewards(self,node,policy_module):
        global precomputed_gamma

        flattened_state = torch.from_numpy(node.stateRepresentation.flatten())


        state_bool=np.array(node.stateRepresentation,dtype=bool)
        valid_predicates_np = state_bool.flatten()   # function calculating the mask requrired to convey to the Probabilistic actor which actions are available

        for _pred_id, pred in enumerate(self.predicatesList):           # masking out infeasible actions that may occur due to underfilled pages.
            if valid_predicates_np[_pred_id]:
                leftNumPoints = (node.pointsArray[:,pred[0]]<pred[1]).sum()
                rightNumPoints = node.pointsArray.shape[0]-leftNumPoints
                if not (leftNumPoints>(self.PageSize*page_frac) and rightNumPoints>(self.PageSize*page_frac)):
                    valid_predicates_np[_pred_id] = False

        node.mask=valid_predicates_np
        valid_predicates = torch.from_numpy(valid_predicates_np)

        
        # assume it is a leaf node and compute its skipping ratio and reward
        # node.queryOverlapCount = 0
        # for q in self.queries:
        #     if max(q[0],node.mbr[0]) < min(q[2],node.mbr[2]) and  max(q[1],node.mbr[1]) < min(q[3],node.mbr[3]):
        #         node.queryOverlapCount+=1
        
        node.queryOverlapCount = node_query_overlap_count(self.queries,*node.mbr)             # the cython magic for speed.

        node.S = (self.queries.shape[0] - node.queryOverlapCount) * node.numberOfPoints
        node.reward = node.S/(self.queries.shape[0] * node.numberOfPoints)      # if this node is split this will be recomputed

        # Check if the node could/needs be further split
        if node.pointsArray.shape[0] > self.PageSize and  torch.any(valid_predicates): 
            self.policy_tuple = policy_module(flattened_state,mask=valid_predicates)
            if self.policy_tuple[-2].numel() == 1:
                predicate_id =  (self.policy_tuple[-2]).item()
            else:
                predicate_id = (self.policy_tuple[-2].argmax()).item() # extracting the numerical action from policy_tuple



            ################ Finding a valid split for the node ######################
            
            
            tries = 10    # Sometimes the policy module returns an invalid split (one child is tiny), in such cases try few times.
            while tries>0:

                if(not valid_predicates[predicate_id]): # Debugging assert
                    print('state',flattened_state)
                    print('mask',valid_predicates_np)
                    print('predicate_id',predicate_id)

                pred = self.predicatesList[predicate_id]
                leftChildPointsArray = node.pointsArray[node.pointsArray[:,pred[0]]<pred[1]]
                rightChildPointsArray = node.pointsArray[node.pointsArray[:,pred[0]]>pred[1]]

                # if they are acceptable
                if leftChildPointsArray.shape[0]>(self.PageSize*page_frac) and rightChildPointsArray.shape[0]>(self.PageSize*page_frac):

                    node.leftChild = QdNode(node.stateRepresentation,leftChildPointsArray)
                    node.rightChild = QdNode(node.stateRepresentation,rightChildPointsArray)
                    node.predicate = predicate_id
                    node.isLeaf = False
                    node.pointsArray = None
                    del leftChildPointsArray, rightChildPointsArray
                    
                    if pred[0]==0:
                        for i in range(predicate_id,(ActionSpaceSize//2)):
                            node.leftChild.stateRepresentation[0,i] = 0
                        node.leftChild.mbr = node.mbr.copy()
                        node.leftChild.mbr[2]=pred[1]

                        for i in range(0,predicate_id+1):
                            node.rightChild.stateRepresentation[0,i] = 0
                        node.rightChild.mbr = node.mbr.copy()
                        node.rightChild.mbr[0]=pred[1]

                    if pred[0]==1:
                        predicate_id-=(ActionSpaceSize//2)
                        for i in range(predicate_id,(ActionSpaceSize//2)):
                            node.leftChild.stateRepresentation[1,i] = 0
                        node.leftChild.mbr = node.mbr.copy()
                        node.leftChild.mbr[3]=pred[1]

                        for i in range(0,predicate_id+1):
                            node.rightChild.stateRepresentation[1,i] = 0
                        node.rightChild.mbr = node.mbr.copy()
                        node.rightChild.mbr[1]=pred[1]

                    self.recursivelySplitAndCalculateRewards(node.leftChild,policy_module)
                    self.recursivelySplitAndCalculateRewards(node.rightChild,policy_module)


                    # reward in the paper.
                    node.S = (node.leftChild.S + node.rightChild.S) 
                    node.reward = node.S/(self.queries.shape[0] * node.numberOfPoints)

                    # reward suggest by Michael. To compute the reward of an action as the reward computed with the new splits as a leaf nodes.
                    node.oneLevelReward = ((self.queries.shape[0] - node.leftChild.queryOverlapCount) * node.leftChild.numberOfPoints + (self.queries.shape[0] - node.rightChild.queryOverlapCount) * node.rightChild.numberOfPoints) / (self.queries.shape[0] * node.numberOfPoints)

                    # Attempt at different reward paradigm computing the discounted bottom up reward.
                    node.precomputedValueTarget = node.reward + (node.leftChild.reward + node.rightChild.reward) *precomputed_gamma

                    return   # break from the while loop, since you have found a split the node.

                else:       # try again to find a new split.
                    self.policy_tuple = policy_module(flattened_state,mask=valid_predicates)
                    if self.policy_tuple[-2].numel() == 1:
                        predicate_id =  (self.policy_tuple[-2]).item()
                    else:
                        predicate_id = (self.policy_tuple[-2].argmax()).item() # extracting the numerical action from policy_tuple                    
                    tries-=1
        return




    def trainQDTreeCreateEpisodicStateActionReward(self,policy_module, num_trajectories = num_traectories_per_iter):
        """
        An all encompassing function to:
            - create a qd-tree by stepping through the decisions to split each node
            - Calculate and propagate up the rewards from the child nodes
            - Collect the required tensordict data  
        """
        list_of_tuples = [] #state(obs), 'hidden', 'logits', 'action', 'sample_log_prob', reward, next-state, done , mask, next-state-mask 

        sum_rootreward_over_trajectories = 0
        sum_average_node_depth = 0
        best_reward = -np.inf
        best_root = None
        for _ in range(num_trajectories): 
            _ = self.reset()
            self.recursivelySplitAndCalculateRewards(self.root,policy_module)


            # collect all tuples from tree.
            self.explorationQueue.append(self.root)
            while len(self.explorationQueue)>0:
                curr_node = self.explorationQueue.popleft()
                
                if not curr_node.leftChild.isLeaf:
                    self.explorationQueue.append(curr_node.leftChild)
                if not curr_node.rightChild.isLeaf:
                    self.explorationQueue.append(curr_node.rightChild)


                next_state = np.zeros((ActionSpaceSize,),dtype=np.single)
                done_flag = True
                if len(self.explorationQueue)>0:
                    next_state = self.explorationQueue[0].stateRepresentation.flatten()
                    done_flag = False

                next_state_mask = next_state.astype(bool)

                
                list_of_tuples.append(
                    (
                        torch.from_numpy(curr_node.stateRepresentation.flatten()),  # observation
                        self.policy_tuple[0],  # hidden
                        self.policy_tuple[1],  # logits
                        self.policy_tuple[2].unsqueeze(0),  # action
                        self.policy_tuple[3],  # sample_log_prob
                        torch.tensor(
                            [curr_node.oneLevelReward],
                        ),  # reward
                        torch.from_numpy(next_state),  # next_observation
                        torch.tensor(
                            [done_flag],
                            dtype=torch.bool,
                        ), # done
                        torch.from_numpy(curr_node.mask), # mask
                        torch.from_numpy(next_state_mask),# next mask
                    )
                )

            if (curr_reward:=self.root.reward) > best_reward:
                best_reward = curr_reward
                best_root = copy.deepcopy(self.root)
            sum_rootreward_over_trajectories+= self.root.reward
            sum_average_node_depth += self.averageNodeDepth()

        ############## rearranging the list of tuples into a tensordict ##################
        # random.shuffle(list_of_tuples)
        t_states = torch.stack([t[0] for t in list_of_tuples])
        t_hidden = torch.stack([t[1] for t in list_of_tuples])
        t_logits = torch.stack([t[2] for t in list_of_tuples])
        t_action = torch.stack([t[3] for t in list_of_tuples])
        t_sampl_log_prob = torch.stack([t[4] for t in list_of_tuples])
        t_rewards = torch.stack([t[5] for t in list_of_tuples])
        t_next_state = torch.stack([t[6] for t in list_of_tuples])
        t_done = torch.stack([t[7] for t in list_of_tuples])
        t_mask = torch.stack([t[8] for t in list_of_tuples])
        t_next_mask = torch.stack([t[9] for t in list_of_tuples])

        t_sampl_log_prob = t_sampl_log_prob.detach()

        td = TensorDict(
            {
                "action": t_action,
                "hidden": t_hidden,
                "next": {
                    "done": t_done,
                    "observation": t_next_state,
                    "reward": t_rewards,
                    "mask": t_next_mask,
                },
                "observation": t_states,
                "sample_log_prob": t_sampl_log_prob,
                "logits": t_logits,
                "mask": t_mask,
            },
            batch_size=t_states.shape[0],
        )

        return td , sum_rootreward_over_trajectories/num_traectories_per_iter, sum_average_node_depth/num_traectories_per_iter, best_reward, best_root

    def reset(self):
        """
        Traverse through the Qd-tree and delete all nodes.
        """
        def recursivelyDelete(node):
            if node is None:
                return 

            if node.isLeaf == False:
                recursivelyDelete(node.leftChild)
                recursivelyDelete(node.rightChild)    
            del node

        recursivelyDelete(self.root)
        self.root = QdNode(np.ones((2,ActionSpaceSize//2),dtype=np.float32),self.pointsList)
        self.root.mbr = [0.0,0.0,1.0,1.0]
        self.explorationQueue.clear()
        self.currentNode = self.root
        return self.root.stateRepresentation

    def saveQdTree(self,filename):

        with open(filename, 'w') as fileoperator:
            def recursivelySaveQdTree(node,node_id):
                fileoperator.write('{} {} {}\n'.format(node_id,self.predicatesList[node.predicate][0],self.predicatesList[node.predicate][1]))

                if not node.leftChild.isLeaf:
                    recursivelySaveQdTree(node.leftChild, (node_id*2)+1)
                if not node.rightChild.isLeaf:
                    recursivelySaveQdTree(node.rightChild, (node_id*2)+2)
            # if at least one cut is performed
            if self.root.predicate is not None:
                recursivelySaveQdTree(self.root,0)            

        # fig,ax = plt.subplots()
        # for q in self.queries:
        #     ax.add_patch(Rectangle((q[0],q[1]),q[2]-q[0],q[3]-q[1],fill=False,alpha=0.5,ec='r'))   
        # def recursivelyPlotSPlits(node):
        #     pred = self.predicatesList[node.predicate]

        #     if pred[0] == 0:
        #         ax.plot([pred[1],pred[1]],[node.mbr[1],node.mbr[3]],c='blue')

        #     elif pred[0] == 1:
        #         ax.plot([node.mbr[0],node.mbr[2]],[pred[1],pred[1]],c='blue')
        #     else:
        #         assert(False)

        #     if not node.leftChild.isLeaf:
        #         recursivelyPlotSPlits(node.leftChild)
        #     if not node.rightChild.isLeaf:
        #         recursivelyPlotSPlits(node.rightChild)
        # if self.root.predicate is not None:
        #     recursivelyPlotSPlits(self.root)
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        # plt.title(f"{self.root.reward:.5}")
        # plt.savefig(filename[:-4]+'.png')
        # plt.close()

    def averageNodeDepth(self):

        def averageNodeDepthRecursively(node,depth=0):
            if node is None:
                return 0.0,0.0

            a,b,c,d=0.0,0.0,0.0,0.0

            if node.isLeaf == False:
                a,b = averageNodeDepthRecursively(node.leftChild,depth+1)
                c,d = averageNodeDepthRecursively(node.rightChild,depth+1)

            return a+c+depth,b+d+1

        sum_depth, num_nodes = averageNodeDepthRecursively(self.root)
        return sum_depth/num_nodes 

    def draw_data_queres(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        for q in self.queries:
            ax.add_patch(Rectangle((q[0],q[1]),q[2]-q[0],q[3]-q[1],fill=False,alpha=0.5,ec='r'))   
        ax.scatter(self.pointsList[:,0],self.pointsList[:,1],s=1)
        # for q in self.predicatesList[:ActionSpaceSize//2]:
        #     ax.axvline(q[1],color="gray")
        # for q in self.predicatesList[ActionSpaceSize//2:]:
        #     ax.axhline(q[1],color="gray")
        return ax
