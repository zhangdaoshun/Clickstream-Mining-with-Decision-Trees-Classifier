import numpy as np
import random
import argparse, os, sys
import pandas as pd
import copy
import time
import csv
import pickle as pkl
from scipy.stats import chisquare

'''
TreeNode represents a node in the decision tree
TreeNode can be a leaf node or a non-leaf node:
    non-leaf node: 
        - data: contains the feature number using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter.

'''

sys.setrecursionlimit(100000)
internal_nodes = 0
leaves = 0



class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)


       
def build_tree(sample_feature, features, pvalue):
    global internal_nodes, leaves
      
    if (sample_feature['output_value'] == 1).sum() == sample_feature['output_value'].count():
        leaves += 1
        return TreeNode()        
    
    if (sample_feature['output_value'] == 0).sum() == sample_feature['output_value'].count():
        leaves += 1
        return TreeNode('F')
    # if there are no features, build the node with positive or negative
    if len(features) == 0:
        true = 0
        false = 0
        true = (sample_feature['output_value'] == 1).sum()
        false = (sample_feature['output_value'] == 0).sum()
        if true >= false:
            leaves += 1
            return TreeNode()            
        else:
            leaves += 1
            return TreeNode('F')        
    # chose the best feature
    chosen_feature = find_best_feature(sample_feature, features)    

    features.remove(chosen_feature)
    node = None
   
    chisq = chisquare_criterion(sample_feature, chosen_feature) 
    # build the node if the chi sqaure value if less than p_value, else terminate   
    if chisq < pvalue:        
        node = TreeNode(chosen_feature + 1)
        internal_nodes += 1
        values = sample_feature[chosen_feature].unique()
        i = 1
        true_missing_value = -1
        false_missing_value = -1
        while i < 6:
            # build the child nodes 
            if i in values:
                sample_feature_subset = sample_feature.loc[sample_feature[chosen_feature] == i]
                if sample_feature_subset.empty:
                    true = (sample_feature['output_value'] == 1).sum()
                    false = (sample_feature['output_value'] == 0).sum()
                    if true >= false:
                        leaves += 1
                        node.nodes[i - 1]= TreeNode()
                    else:
                        leaves += 1
                        node.nodes[i - 1] = TreeNode('F')
                else:       
                    attri = copy.deepcopy(features)                                
                    is_node = build_tree(sample_feature_subset, features, pvalue)
                    if is_node:
                        node.nodes[i - 1] = is_node
                    else:
                        true = (sample_feature_subset['output_value'] == 1).sum()
                        false = (sample_feature_subset['output_value'] == 0).sum()
                        if true >= false:
                            leaves += 1
                            node.nodes[i - 1]= TreeNode()
                        else:
                            leaves += 1
                            node.nodes[i - 1] = TreeNode('F')
            else:
                if true_missing_value == -1 and false_missing_value == -1:
                    true_missing_value = (sample_feature['output_value'] == 1).sum()
                    false_missing_value = (sample_feature['output_value'] == 0).sum()
                if true_missing_value >= false_missing_value:
                    leaves += 1
                    node.nodes[i - 1] = TreeNode()
                else:
                    leaves += 1
                    node.nodes[i - 1] = TreeNode('F')
            i += 1
    else:                       
        return None
    return node    

# traverse and return the best possible value        
def classify(root, datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return classify(root.nodes[datapoint[int(root.data)-1]-1], datapoint)  

def chisquare_criterion(sample_feature, chosen_feature):    
    observed = []
    expected = []   
    n = (sample_feature['output_value'] == 0).sum()
    p = (sample_feature['output_value'] == 1).sum()
    N = n + p
    values = sample_feature[chosen_feature].unique()    
    # for each value calculate the expected and observed number of positives and negatives
    for value in values:                
        fsample_feature = sample_feature.filter([chosen_feature,'output_value'],axis=1)
        fsample_feature = fsample_feature.loc[(fsample_feature[chosen_feature]==value)]
        T_i = fsample_feature['output_value'].count()                                        
        p_prime_i = float(float(p) / N) * T_i
        n_prime_i = float(float(n) / N) * T_i
        p_i = float((fsample_feature['output_value'] == 1).sum())
        n_i = float((fsample_feature['output_value'] == 0).sum())        
        if p_prime_i != 0:
            expected.append(p_prime_i)
            observed.append(p_i)
        if n_prime_i != 0:
            expected.append(n_prime_i)        
            observed.append(n_i)
    c, p = chisquare(observed, expected)          
    return p

# get the best feature from the remaining features 
def find_best_feature(sample_feature, features):    
    min_entropy = None
    best_feature = None    
    # calculate the gain and the feature with maximum gain
    for feature in features:                 
        rows_count = sample_feature[feature].count()
        values = sample_feature[feature].unique()        
        entropy_value = 0    
        # calculate the gain and add all the gain values
        for value in values:
            count = (sample_feature[feature] == value).sum()
            p = float(count) / rows_count            
            fsample_feature = sample_feature.filter([feature, 'output_value'], axis=1)            
            fsample_feature = fsample_feature.loc[(fsample_feature[feature] == value)]
            fsample_feature_rows_count = fsample_feature['output_value'].count()            
            true = (fsample_feature['output_value'] == 1).sum()
            prob_true = float(true) / fsample_feature_rows_count
            false = (fsample_feature['output_value'] == 0).sum()
            prob_false = float(false) / fsample_feature_rows_count
            if prob_true == 0:
                entropy_true = 0
            else:
                entropy_true = prob_true * (np.log2(prob_true))
            if prob_false == 0:
                entropy_false = 0
            else:
                entropy_false = prob_false * (np.log2(prob_false))                            
            total_entropy = -(entropy_false + entropy_true)        
            entropy_value += p * total_entropy
        if min_entropy == None or entropy_value < min_entropy:
            best_feature = feature
            min_entropy = entropy_value 
    # return the best feature   
    return best_feature

# parse the command line arguments    
parser = argparse.ArgumentParser()
parser.add_argument('-p', help='specify p-value', dest='pvalue', action='store', default='0.005')
parser.add_argument('-f1', help='specify training dataset', dest='train_dataset', action='store', default='')
parser.add_argument('-f2', help='specify test dataset', dest='test_dataset', action='store', default='')
parser.add_argument('-o', help='specify output file', dest='output_file', action='store', default='')
parser.add_argument('-t', help='specify decision tree', dest='decision_tree', action='store', default='') 
   
args = vars(parser.parse_args())


train_data_file_name = args['train_dataset']
train_data_label_file = args['train_dataset'].split('.')[0] + '_label.csv'

sample_feature = pd.read_csv(train_data_file_name, header=None, sep=" ")    
train_output_values = pd.read_csv(train_data_label_file, header=None)
sample_feature['output_value'] = train_output_values[0]
feature_count = sample_feature.shape[1] - 1
features = [i for i in range(feature_count)]
pvalue = float(args['pvalue'])

# build the tree
root = build_tree(sample_feature, features, pvalue)   
root.save_tree(args['decision_tree'])  


test_data_file = args['test_dataset']
test_data_space = pd.read_csv(test_data_file, header=None, sep=" ")
test_row_count = test_data_space.shape[0]
test_result = []

# get the best possible output value for the test data
for i in range(test_row_count):
    test_result.append([classify(root, test_data_space.loc[i])])

# write the test output
output_file = args['output_file']
with open(output_file, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(test_result)
print("internal nodes: ", internal_nodes, " leaf nodes: ", leaves)






