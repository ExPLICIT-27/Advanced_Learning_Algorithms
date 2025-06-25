import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
_ = plot_entropy()
"""
using the cat classifier example, with one hot encoding
ear shape, face shape and whiskers
"""
X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p*np.log2(p) - (1 - p)*np.log2(1 - p)

print(entropy(0.5))

def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
    that feature = 1 and the right node those that have the feature = 0 
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[index_feature]:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices
split_indices(X_train, 0)

def weighted_entropy(X, y, left_indices, right_indices):
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = entropy(w_left)
    p_right = entropy(w_right)
    weighted_entropy = w_left*p_left + w_right*p_right
    
    return weighted_entropy

left_indices, right_indices = split_indices(X_train, 0)
print(left_indices, right_indices)
print(weighted_entropy(X_train, y_train, left_indices, right_indices))

def information_gain(X, y, left_indices, right_indices):
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy

print(information_gain(X_train, y_train, left_indices, right_indices))

for i, feature_name in enumerate(['Ear shape', 'Face shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")

tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=1, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)
tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=2, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)