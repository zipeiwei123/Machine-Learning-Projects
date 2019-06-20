from RandomForest import RandomForest
from main import load_data
from functions import *


import unittest
class test_decisionTree(unittest.TestCase):
    def test_DT(self):
        records, attributes = load_data("data/mushrooms_train.data")
        test_records = load_data("data/mushrooms_train.data")[0]
        #print(records, attributes)
        RF = RandomForest(tree_num = 10)

        RF.train(records, attributes)
    	
    
    	# correct_cnt = 0
    	# for sample in test_records:
     #    	if RF.predict(sample) == sample["label"]:
     #        	correct_cnt += 1
    	# print ("Accuracy:",float(correct_cnt) / len(test_records))

        
        
    

if __name__ == "__main__":
	unittest.main()