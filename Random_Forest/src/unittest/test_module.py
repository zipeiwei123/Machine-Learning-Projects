from __future__ import print_function, division
from DecisionTree import DecisionTree, TreeNode
from main import load_data
import sys
from pprint import pprint

import unittest


class test_decisionTree(unittest.TestCase):

    def test_DT(self):
        records, attributes = load_data("data/mushrooms_train.data")
        test_records = load_data("data/mushrooms_train.data")[0]
        #print(records, attributes)
        dt = DecisionTree()
        best_index, best_index_dict = dt.find_best_split(records, attributes, class_index = 0)
        dt.shuffle_dataset(best_index_dict)

        # SUCCESS!!!!!


if __name__ == "__main__":
    unittest.main()
