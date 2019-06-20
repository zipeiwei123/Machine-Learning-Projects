from __future__ import print_function, division
from random import randrange, sample
from functions import *
"""import all the functions from decision trees"""
from DecisionTree import *

class RandomForest(object):
    """
    Class of the Random Forest
    """
    def __init__(self, tree_num):
        self.tree_num = tree_num
        self.forest = []
        self.get_tree = []

    """randomly selected 50 % of the attribute and use it to build the forest"""
    def shuffle_attributes(self, attributes, half):
        unique = set()
        while len(unique) < half:
            unique = sample((attributes), int(half))
        return unique
        

    def train(self, records, attributes):
        """Create subsample for each tree """
        self.tree_num = int(self.tree_num)
        for number in range(self.tree_num):
            self.forest.append(self.bootstrap(records))
        half = len(attributes)/2
        index = 1
        for tree in self.forest:
            dt = DecisionTree()
            print("Create TREE %d\n"%(index))
            attributes = self.shuffle_attributes(attributes, half)
            dt.train_random_forest(tree, attributes)
            self.get_tree.append(dt)
            index += 1
  
    """get the predict value of random forest"""
    def predict(self, sample):
        prediction_list = list()
        e_counter = 0
        p_counter = 0
        i = 0
        for tree in self.forest:
            i+= 1
            index = self.forest.index(tree)
            label = self.get_tree[index].predict(sample)
            prediction_list.append(label)
            index += 1
        for value in prediction_list:
            if value == 'p':
                p_counter += 1
            elif value == 'e':
                e_counter += 1
        if p_counter >= e_counter:
            return 'p'
        else:
            return 'e'
     
    """create sample with replacement that equals original dataset"""
    def bootstrap(self, records):
        sample = list()
        for record in records:
            index = randrange(len(records))
            sample.append(records[index])
        return sample



        

        
