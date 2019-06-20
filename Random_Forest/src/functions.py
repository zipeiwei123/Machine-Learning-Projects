from __future__ import print_function, division
import math
from random import randrange
from collections import defaultdict, Counter
from pprint import pprint

"""Check if in the node, all the label belongs to one class""" 
def one_class_only(records):
    number_of_instance = len(records)
    p_counter = 0
    e_counter = 0
    for record in records:
        if record["label"] == 'p':
            p_counter += 1
        elif record["label"] == 'e':
            e_counter += 1
    if p_counter == number_of_instance or e_counter == number_of_instance:
        return True
    else:
        return False
        
"""Split current dataset by each unique attribute value == subset that contains unique attribute value"""
def split_instances(instances, attribute_index):
    partitions = defaultdict(list)
    for instance in instances:
        partitions[instance["attributes"][attribute_index]].append(instance)
    return partitions

"""find the most common class label for each node"""
def choose_best_attribute_index(instances, candidate_attribute_indexes, class_index=0):
    gains_and_indexes = sorted([(information_gain(instances, i), i) for i in candidate_attribute_indexes], 
                           reverse=True)
    return gains_and_indexes[0][1]

"""Get information gain for each ndoe"""
def information_gain(instances, parent_index, class_index=0, attribute_name=False):

    parent_entropy = entropy(instances, class_index, attribute_name)
    child_instances = defaultdict(list)
    for instance in instances:
        child_instances[instance['attributes'][parent_index]].append(instance)
    children_entropy = 0.0
    num_instances = len(instances)
    for child_value in child_instances:
        child_probability = len(child_instances[child_value]) / num_instances
        children_entropy += child_probability * entropy(
            child_instances[child_value], class_index, attribute_name, child_value)
    return parent_entropy - children_entropy

"""Calculate entropy for each node"""
def entropy(instances, class_index=0, attribute_name=None, value_name=None):
    num_instances = len(instances)
    if num_instances <= 1:
        return 0
    value_counts = defaultdict(int)
    for instance in instances:
        value_counts[instance['label']] += 1
    num_values = len(value_counts)
    if num_values <= 1:
        return 0
    attribute_entropy = 0.0
    if attribute_name:
        print('entropy({}{}) = '.format(attribute_name, 
            '={}'.format(value_name) if value_name else ''))
    for value in value_counts:
        value_probability = value_counts[value] / num_instances
        child_entropy = value_probability * math.log(value_probability, 2)
        attribute_entropy -= child_entropy
        if attribute_name:
            print('  - p({0}) x log(p({0}), {1})  =  - {2:5.3f} x log({2:5.3f})  =  {3:5.3f}'.format(
            value, num_values, value_probability, child_entropy))
    if attribute_name:
        print('  = {:5.3f}'.format(attribute_entropy))
    return attribute_entropy

"""return the most common label for a tree prediction list"""
def most_common(prediction_list):
    e_counter = 0
    p_counter = 0
    for value in prediction_list:
        
        if value == 'e':
            e_counter = e_counter + 1
        elif value == 'p':
            p_counter = p_counter + 1
    
    if p_counter >= e_counter:
        return 'p'
    elif p_counter < e_counter:
        return 'e'

"""Countes the attributes"""
def attributes_counts(records, index):
    attribute_counts = list()
    for record in records:
        if record["attributes"][index] not in attribute_counts:
            attribute_counts.append(record["attributes"][index])
    return attribute_counts

"""shuffle the dataset with only upload 75% of the original dataset"""
def shuffle_dataset(best_index_dict):
    partitions = defaultdict(list)
    for value in best_index_dict:
        get_length = int(len(best_index_dict[value])* 0.75)
        for index in range(get_length):
            index = randrange(get_length)
            partitions[value].append(best_index_dict[value][index])
    return partitions
