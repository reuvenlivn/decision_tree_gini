# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:26:30 2019

@author: reuve
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#reading data and pre-processing
file_name = 'wdbc.data'
dataset = pd.read_csv(file_name)

#y = f(x) is it M or B
y = dataset['M']
# x are all the features (without the y)
x = dataset.drop('M',axis=1)
# array of 1 and 0 to split the dataset to test and train set
y_float = (y=='M').astype(float)
#create the train and test sets
x_train,x_test,y_train,y_test = train_test_split(x, y_float, train_size=0.8)
# train array
train_data = np.column_stack((x_train, y_train))
#test array
test_data = np.column_stack((x_test, y_test))

#1
#a
#split the tree .
#input: the datatset, index of the feature and value
#output: left and right branches
def split_node(dataset, index, value):
    left = []
    right = []
    for row in dataset:
        if row[index]>value:
            right.append(row)
        else:
            left.append(row)
    return np.array(left),np.array(right)

#b    
#calcualte Gini value
#input: left and right nodes
#output gini value
def get_gini(left,right):
    # init gini and total_size
    gini = 0
    total = len(left) + len(right)
    # go over the two branches
    for branch in [left,right]:
        #init the branch score and size
        score = 0
        size = len(branch)
        # verify the branch is not empty
        if size==0:
            continue
        # go over all the labels 
        for label in [0,1]:
                #calc prob: count number of instances for the label
            prob = [raw[-1] for raw in branch].count(label)/size
                #sum the score
            score += prob*prob
        # update gini value
        gini += (1-score)*size/total
    return gini

#find the best feature (and best value) for next split
#input: dataset including the label
def get_split(dataset):
    # save number of raws and colums
    num_rows = dataset.shape[0]
    num_columns = dataset.shape[1] 
    # init gini to be max integer
    gini = float('inf')
    
    for index in range(num_columns-1):
        for row in range(num_rows):
            # make a node split
            left,right = split_node(dataset, index, dataset[row, index])
            # calculate thr gini index
            g = get_gini(left,right)
            # better gini?
            if g < gini:
                # save the best gini (the minimal)
                gini = g
                best_feature = index
                best_value = dataset[row, index]
                best_right = right
                best_left = left
    # return the best values
    return best_feature,best_value,best_right,best_left,gini

#c
max_depth = 6

class Node(object):    
    def __init__(self, dataset, depth):
        self.dataset = dataset
        self.current_depth = depth
        self.right = None
        self.left = None
        self.feature_index = None
        self.value = None
        self.label = None
     
    # build the tree    
    def build_tree(self):
        # get the label with the highest probebility
        label_prob,self.label = self.majority_label_prob()
        # chack if end of tree traversal
        if self.current_depth > max_depth or label_prob >= 1:
            print('leaf')
            return
        else:
            # init label again
            self.label = None
            # split the node
            self.feature_index,self.value,right_dataset,left_dataset,best_gini = get_split(self.dataset)
            print('[{}], {} samples, X{}=={}, gini={}'.
                  format(self.current_depth,len(self.dataset),self.feature_index,self.value,best_gini))                
            # go right
            print('-->')
            self.right = Node(right_dataset,self.current_depth+1)
            self.right.build_tree()
            # then go left
            print('<--')
            self.left = Node(left_dataset,self.current_depth+1)
            self.left.build_tree()
            
    # get the label with the highest probebility           
    def majority_label_prob(self):
        #count zeros
        label_zero = [row[-1] for row in self.dataset].count(0)
        #count ones
        label_one = [row[-1] for row in self.dataset].count(1)
        # save total labels
        total =  label_zero + label_one

        if label_one > label_zero:
            #more ones
            majority = 1
            label_prob = label_one / total
        else:
            # more zeros
            majority = 0
            label_prob = label_zero / total

        return label_prob,majority

            
    def predict(self,row):
       if self.label != None:
           return self.label
       else:
           if row[self.feature_index] < self.value:
               return self.left.predict(row)
           else:
               return self.right.predict(row)
        
#main
#create node object               
root_node = Node(train_data,1)
#build the tree
root_node.build_tree()
#
test_labels = []
for row_data in test_data:
    row_label = root_node.predict(row_data)
    test_labels.append(row_label)

accuracy = 1-sum(abs(test_labels - y_test))/len(y_test)
