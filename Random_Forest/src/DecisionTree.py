from __future__ import print_function, division
from math import log
from random import randrange, sample
from collections import defaultdict, Counter
"""functions.py contains all the calculation functions such as info_gain, entropy"""
from functions import *



class TreeNode(object):
    """Create a treenode object where class_label == 'p' or 'e', 
                                Attribute_index will help train data to locate the attribute index,
                                data is current node's data, 
                                attribute_value is from parent_node -> attribute_index->attribute_value = child,
                                isLeaf default to False """
    def __init__(self, attribute_index, attribute_value, data, class_label = "", isLeaf = False):
        self.class_label = class_label
        self.attribute_index = attribute_index
        self.attribute_value = attribute_value, 
        self.data = data
        self.children = []
        self.isLeaf = isLeaf

    """Call self.root recursively to obtained the class_label on the leaf node where isLeaf = True"""
    def predict(self, sample):
        if self.isLeaf == True:
            return self.class_label
        value = sample['attributes'][self.attribute_index]

        for node in self.children:
            convert_to_string = ''.join(node.attribute_value)
            if value == convert_to_string:
                return node.predict(sample)
        
    """Append the child node to parent node"""
    def add_child(self, node):
        self.children.append(node)

    """Get current_node's number of children"""
    def num_children(self):
        return len(self.children)

    """Calling self.root recursively to obtained all the leaf node on the tree"""
    def num_leaves(self):
        
        if self.isLeaf == True:
            return 1
        else:
            return sum(c.num_leaves() for c in self.children)

    """print the current tree node"""
    def print_node(self):
        return "Current node value is:{0}, Attribute_index: {1}, class_label: {2}, Dataset: {3}, IsLeaf: {4})".format( 
          self.attribute_value, 
          self.attribute_index,
          self.class_label, 
          len(self.data),
          self.isLeaf
          )
 
class DecisionTree(object):
    """
    Class of the Decision Tree
    """

    def __init__(self):
        self.root = None
        
    """Train decision tree model with tree_growth, obtained root that connected all other node"""
    def train(self, records, attributes):
        self.root, split_index = self.tree_growth(records, attributes, parent_node = None,  split_index = list())
        print("The total number leaf nodes of the tree is: %d"%(self.root.num_leaves()))
        print("The split index is", split_index)
        print("Number of children on the root", self.root.num_children())
        print("The tree depth is", len(split_index))
        
        
    """different train function for random forest """
    def train_random_forest(self, records, attributes):
        """get random candidate attributes before build the tree"""
        print("After shuffle, the attributes is %s and length is %d"%(attributes, len(attributes)))
        self.root, split_index = self.tree_growth_random_forest(records, attributes,  parent_node = None, split_index = list())
        print("The total number leaf nodes of the tree is: %d"%(self.root.num_leaves()))
        print("The split index is", split_index)
        print("Number of children on the root", self.root.num_children())
        print("The tree depth is", len(split_index))
    
    """return the sample label, which is either 'p' or 'e"""
    def predict(self, sample):
        return self.root.predict(sample)


    """ Check if the stopping condition is met, stopping condition is met either record's size = 0 or no remaining attributes"""
    def stopping_cond(self, records, attributes):
        if len(records) == 0 or len(attributes) == 0:
            return True
        else:
            return False

    """ Get the node label if we have mixed 'p' and 'e'"""
    def classify(self, records):
        # determine the root node be p or e
        p_counter = 0
        e_counter = 0
        for record in records:
            if record["label"] == 'p':
                p_counter += 1
            elif record["label"] == 'e':
                e_counter += 1
        if p_counter > e_counter:
            return 'p'
        else:
            return 'e'

    """Get the best split from current node, data, and return the next best attribute, and a dictionary which keys are:
        unique value for the best attribute index, 
        values: the subset that contains the unique attribute value               """
    def find_best_split(self, records, attributes, class_index):
        #get the best index first
        best_index = choose_best_attribute_index(records,  attributes, class_index)
        #create a dictionary based on the best index
        best_index_dict = split_instances(records, best_index)
        return best_index, best_index_dict

   

    """Tree_growth called itself iterative, if the root is None, create root
        else: 
            create child node and connected with root node
            if stopp condition met, set isLeaf = False, and stop recusive on this node.
            Otherwise, keep recusion until all the left nodes are leaf node.
            (For more details, please check the ID3 algorithms)
                                                                                        """
    def tree_growth(self, records, attributes, parent_node = None, split_index = ()):
        """trace is use to print the node during tree_growth iteration, by assigning trace = 1 to turn off the print statement"""
        trace = 0
        """if the stop condition met """
        if self.stopping_cond(records, attributes) == True:
            parent_node.isLeaf = True
            return parent_node.isLeaf
        #if the subset has only one class label
        elif one_class_only(records) == True:
            parent_node.class_label  = self.classify(records)
            return parent_node.class_label
        else:
            if parent_node == None:
                #create initial node
                label = self.classify(records)
                self.root = TreeNode(class_label = label, attribute_index = None, attribute_value = None, data = records, isLeaf = False)
                parent_node = self.root
            parent_node.isLeaf = False
            best_index, best_index_dict = self.find_best_split(records, attributes, class_index = 0)
            parent_node.attribute_index  = best_index
            remaining_candidate_attribute_indexes = [i for i in attributes if i != best_index]
            
            """only partition node will go through this loop"""
            if trace == 0:
                print("Creating tree node", parent_node.print_node())
            if best_index not in split_index:
                split_index.append(best_index)

            for value in best_index_dict:
                
                label = self.classify(best_index_dict[value])
                child_node  = TreeNode(class_label = label, attribute_index = None, attribute_value = value, data = best_index_dict[value], isLeaf = True)
                parent_node.add_child(child_node)
                if trace == 0:
                    print("Creating a child node", child_node.print_node())
                self.tree_growth(best_index_dict[value], remaining_candidate_attribute_indexes,  child_node, split_index)
          
        return self.root, split_index

    """below is the function use to train random forest, modification from tree growth by passing:
        Randomly selected 50% of the attribute index and use it to build tree
        Only 75% of dataset passed by each node         """
    def tree_growth_random_forest(self, records, attributes, parent_node = None, split_index = ()):
        """trace is use to print the node during tree_growth iteration, by assigning trace = 1 to turn off the print statement"""
        trace = 1
        iteration = 0
    
        """if the stop condition met"""
        if self.stopping_cond(records, attributes) == True:
            parent_node.isLeaf = True
            return parent_node.isLeaf
        #if the subset has only one class label
        elif one_class_only(records) == True:
            parent_node.class_label  = self.classify(records)
            return parent_node.class_label
        else:
            if parent_node == None:
                #create initial node
                label = self.classify(records)
                self.root = TreeNode(class_label = label, attribute_index = None, attribute_value = None, data = records, isLeaf = False)
                parent_node = self.root
            parent_node.isLeaf = False
            best_index, best_index_dict = self.find_best_split(records, attributes, class_index = 0)

            #shuffle the dataset in the node
            best_index_dict = shuffle_dataset(best_index_dict)
            parent_node.attribute_index  = best_index
            remaining_candidate_attribute_indexes = [i for i in attributes if i != best_index]
            
            
            """only partition node will go through this loop"""
            if trace == 0:
                print("Creating tree node", parent_node.print_node())
            if best_index not in split_index:
                split_index.append(best_index)

            for value in best_index_dict:
                label = self.classify(best_index_dict[value])
                child_node  = TreeNode(class_label = label, attribute_index = None, attribute_value = value, data = best_index_dict[value], isLeaf = True)
                parent_node.add_child(child_node)
                if trace == 0:
                    print("Creating a child node", child_node.print_node())
                self.tree_growth_random_forest(best_index_dict[value], remaining_candidate_attribute_indexes, child_node, split_index)
          
        return self.root, split_index