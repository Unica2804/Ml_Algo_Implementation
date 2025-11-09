import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of the feature to split on
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Class label for leaf nodes



class Decisiontree:
    def __init__(self, min_samples_split=2, max_depth=100,categorical_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.cat_cols=set(categorical_features or [])
        self.root = None

    def fit(self,X,y):
        self.root=self._build_tree(X,y)

    
    
    
    def _build_tree(self,X,y,depth=0):
        num_samples, num_features = X.shape # num_samples: rows, num_features: columns
        num_labels = len(np.unique(y))# unique labels in y

        if (depth>=self.max_depth) or (num_labels==1) or (num_samples<self.min_samples_split): # stopping criteria
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        feat_idxs=np.random.choice(num_features, num_features, replace=False)
        best_feature,best_threshold=self._best_split(X,y,feat_idxs)
        true_idxs,false_idxs=self._split(X[:,best_feature],best_threshold)
        left_subtree=self._build_tree(X[true_idxs,:],y[true_idxs],depth+1)
        right_subtree=self._build_tree(X[false_idxs,:],y[false_idxs],depth+1)
        return Node(best_feature,best_threshold,left_subtree,right_subtree)
    


    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    
    def _most_common_label(self,labels):
        counter=Counter(labels)
        most_common=counter.most_common(1)[0][0]
        return most_common


    def _is_numeric(self,value):
        return isinstance(value,int) or isinstance(value,float)
    
    
    def _best_split(self,X,y,feat_idxs):
        best_gain=-1
        split_idx,best_threshold=None,None

        for feature in feat_idxs:
            feat_values=X[:,feature]

            if feature in self.cat_cols:
                cat=np.unique(feat_values)
                
                for c in cat:
                    true_idxs,false_idxs=self._split(feat_values,c)
                    if len(true_idxs)==0 or len(false_idxs)==0:
                        continue
                    gain=self._information_gain(y,true_idxs,false_idxs)
                    if gain>best_gain:
                        best_gain=gain
                        split_idx=feature
                        best_threshold=c
            else:
                threshold=np.unique(feat_values)
                for thr in threshold:
                    true_idxs,false_idxs=self._split(feat_values,thr)
                    if len(true_idxs)==0 or len(false_idxs)==0:
                        continue
                    gain=self._information_gain(y,true_idxs,false_idxs)
                    if gain>best_gain:
                        best_gain=gain
                        split_idx=feature
                        best_threshold=thr
        return split_idx,best_threshold


    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)
    
    
    def _information_gain(self,y,left_idxs,right_idxs):
        current_uncertainty=self._gini(y)
        p=float(len(left_idxs))/len(y)
        gain=current_uncertainty - p*self._gini(y[left_idxs]) - (1-p)*self._gini(y[right_idxs])
        return gain
    

    def _split(self,features,threshold):
        
        """
        input:
        features: Takes in values of feature to split on
        
        threshold: value to compare against

        returns:
        Returns index of samples that satisfy the condition and those that do not

        true_idxs,false__idxs

        """
        if self._is_numeric(threshold):
            left_mask = features >= threshold
        else:
            left_mask = np.isin(features, threshold)
        true_idxs = np.where(left_mask)[0]
        false_idxs = np.where(~left_mask)[0]
        return true_idxs, false_idxs
    

    def _traverse_tree(self, x, node):
        if node.value is not None:                 
            return node.value

        # decide left/right with the SAME rule used in _split
        if self._is_numeric(node.threshold):
            go_left = x[node.feature] >= node.threshold
        else:                                      
            go_left = x[node.feature] in node.threshold   

        if go_left:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

