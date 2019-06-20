
##This project is written in python version 3.6, and re-modified later on to suit python 2.x
# functions.py is a file that contains list of functions that shared both in DecisionTree.py and RandomForest.py

functions.py includes:
    1) GET ENTROPY FOR CURRENT NODE
    2) CALCULATE INFO GAIN FOR CURRENT NODE
    3) SPLIT THE RECORDS BASED ON OBTAINED THE BEST ATTRIBUTE BY CALCULATE ALL THE CHILD NODE'S INFO GAIN
THE DATA STRUCTURE LOOKS LIKE THIS: FOR ATTRIBUTE A: UNIQUE VALUE 1 OF A-> DATASET CONTAINS UNIQUE VALUE 1
                                                     UNIQUE VALUE 2 OF A-> DATASET CONTAINS UNIQUE VALUE 2
                                                     UNIQUE VALUE 3 OF A-> DATASET CONTAINS UNIQUE VALUE 3

IN DECISIONTREE.PY
4) FOR THE NODE STURCUTRE:: CLASS_LABEL: LABEL WE OBTAINED FROM CURRENT NODE ('P' OR 'E')

                            ATTRIBUTE_INDEX: ATTRIBUTE INDEX IS STORED IN THE PARENT NODE. FOR INSTANCE, FROM ROOT NODE, WE GET THE NEXT BEST SPLIT IS ATTRIBUTE 5 (WHICH IS 4 IN 0-22 RANGES), ROOT NODE WILL CONTAIN ATTRIBUTE 5. IF THE NODE ATTRIBUTE INDEX == NONE, IT MEANS IT DOESN'T HAVE THE CHILD NODE, AND THE ISLEAF == TRUE

                            ATTRIBUTE_VALUE: BY OBTAINING THE ATTRIBUTE INDEX FROM THE PARENT NODE, WE USE ATTRIBUTE_VALUE TO GET ITS SPECIFIC CHILD NODE WITH THAT VALUE:   SUCH AS: ROOT->'P'->CHILD1

                            CHILDEN:NUMBER OF CHILDREN ON EACH NODE, IT IS A LIST TYPE

                            ISLEAF: DETERMINE IF CURRENT LEAF IS LEAF OR INTERNAL NODE
IN TREE_GROWTH: TRACE = 0, WILL PRINT EACH NODE VALUE AND ISLEAF OR NOT, BY USING TRACE = OTHER INTEGER TO TURN OFF THE PRINT FUNCTION.
We use the ID3 algorithm to construct the tree. For more detail, please visit: http://www.ke.tu-darmstadt.de/lehre/archiv/ws0809/mldm/dt.pdf

OUPUT FOR DECISION TREE:
(py27) zipeis-MacBook-Pro:test zipeiwei$ python src/main.py -m0 -t data/mushrooms_train.data -e data/mushrooms_test.data 
Testing Decision Tree model
Creating tree node Current node value is:(None,), Attribute_index: 4, class_label: e, Dataset: 5694, IsLeaf: False)
Creating a child node Current node value is:('a',), Attribute_index: None, class_label: e, Dataset: 279, IsLeaf: True)
Creating a child node Current node value is:('c',), Attribute_index: None, class_label: p, Dataset: 128, IsLeaf: True)
Creating a child node Current node value is:('f',), Attribute_index: None, class_label: p, Dataset: 1506, IsLeaf: True)
Creating a child node Current node value is:('m',), Attribute_index: None, class_label: p, Dataset: 24, IsLeaf: True)
Creating a child node Current node value is:('l',), Attribute_index: None, class_label: e, Dataset: 279, IsLeaf: True)
Creating a child node Current node value is:('n',), Attribute_index: None, class_label: e, Dataset: 2488, IsLeaf: True)
Creating tree node Current node value is:('n',), Attribute_index: 19, class_label: e, Dataset: 2488, IsLeaf: False)
Creating a child node Current node value is:('b',), Attribute_index: None, class_label: e, Dataset: 33, IsLeaf: True)
Creating a child node Current node value is:('h',), Attribute_index: None, class_label: e, Dataset: 32, IsLeaf: True)
Creating a child node Current node value is:('k',), Attribute_index: None, class_label: e, Dataset: 922, IsLeaf: True)
Creating a child node Current node value is:('o',), Attribute_index: None, class_label: e, Dataset: 35, IsLeaf: True)
Creating a child node Current node value is:('n',), Attribute_index: None, class_label: e, Dataset: 938, IsLeaf: True)
Creating a child node Current node value is:('r',), Attribute_index: None, class_label: p, Dataset: 59, IsLeaf: True)
Creating a child node Current node value is:('w',), Attribute_index: None, class_label: e, Dataset: 438, IsLeaf: True)
Creating tree node Current node value is:('w',), Attribute_index: 21, class_label: e, Dataset: 438, IsLeaf: False)
Creating a child node Current node value is:('g',), Attribute_index: None, class_label: e, Dataset: 189, IsLeaf: True)
Creating a child node Current node value is:('p',), Attribute_index: None, class_label: e, Dataset: 34, IsLeaf: True)
Creating a child node Current node value is:('l',), Attribute_index: None, class_label: e, Dataset: 46, IsLeaf: True)
Creating tree node Current node value is:('l',), Attribute_index: 20, class_label: e, Dataset: 46, IsLeaf: False)
Creating a child node Current node value is:('c',), Attribute_index: None, class_label: p, Dataset: 12, IsLeaf: True)
Creating a child node Current node value is:('v',), Attribute_index: None, class_label: e, Dataset: 34, IsLeaf: True)
Creating a child node Current node value is:('w',), Attribute_index: None, class_label: e, Dataset: 140, IsLeaf: True)
Creating a child node Current node value is:('d',), Attribute_index: None, class_label: p, Dataset: 29, IsLeaf: True)
Creating tree node Current node value is:('d',), Attribute_index: 20, class_label: p, Dataset: 29, IsLeaf: False)
Creating a child node Current node value is:('y',), Attribute_index: None, class_label: e, Dataset: 5, IsLeaf: True)
Creating a child node Current node value is:('v',), Attribute_index: None, class_label: p, Dataset: 24, IsLeaf: True)
Creating a child node Current node value is:('y',), Attribute_index: None, class_label: e, Dataset: 31, IsLeaf: True)
Creating a child node Current node value is:('p',), Attribute_index: None, class_label: p, Dataset: 182, IsLeaf: True)
Creating a child node Current node value is:('s',), Attribute_index: None, class_label: p, Dataset: 398, IsLeaf: True)
Creating a child node Current node value is:('y',), Attribute_index: None, class_label: p, Dataset: 410, IsLeaf: True)
The total number leaf nodes of the tree is: 22
The split index is [4, 19, 21, 20]
Number of children on the root 9
The tree depth is 4
Accuracy: 1.0
(py27) zipeis-MacBook-Pro:test zipeiwei$ 



RANDOM FOREST:

In random forest, I just create 2 extra functions
	one is : shuffle_dataset(): which will return a subset dictionary that each subset is only 75% data of the original. The index is choosed randomly
	second is: shuffle_attributes(): which will return a subset attribute index list with only 50% of the attributes without Replacement. 

	By constructing the random forest, I will get a prediction that range from 0.98~1, which is a pretty good prediction overall.
	

Output for random forest:
(py27) zipeis-MacBook-Pro:test_forest zipeiwei$ python src/main.py -m1 -t data/mushrooms_train.data -e data/mushrooms_test.data -n 20
Testing Random Forest model
Create TREE 1

After shuffle, the attributes is [7, 13, 12, 19, 1, 17, 14, 9, 20, 21, 15] and length is 11
The total number leaf nodes of the tree is: 30
The split index is [19, 7, 20, 21, 1, 9, 17, 12, 14]
Number of children on the root 9
The tree depth is 9
Create TREE 2

After shuffle, the attributes is [7, 21, 17, 20, 14, 1, 12, 15, 13, 19, 9] and length is 11
The total number leaf nodes of the tree is: 25
The split index is [19, 7, 20, 21, 1, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 3

After shuffle, the attributes is [21, 1, 17, 14, 13, 12, 9, 7, 19, 15, 20] and length is 11
The total number leaf nodes of the tree is: 27
The split index is [19, 7, 20, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 4

After shuffle, the attributes is [12, 1, 20, 21, 15, 17, 7, 14, 19, 13, 9] and length is 11
The total number leaf nodes of the tree is: 26
The split index is [19, 7, 20, 21, 1, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 5

After shuffle, the attributes is [20, 14, 13, 1, 7, 12, 19, 15, 17, 9, 21] and length is 11
The total number leaf nodes of the tree is: 29
The split index is [19, 7, 21, 1, 9, 20, 14]
Number of children on the root 9
The tree depth is 7
Create TREE 6

After shuffle, the attributes is [7, 14, 19, 15, 13, 17, 12, 1, 9, 21, 20] and length is 11
The total number leaf nodes of the tree is: 28
The split index is [19, 7, 20, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 7

After shuffle, the attributes is [21, 20, 9, 14, 13, 15, 7, 17, 19, 12, 1] and length is 11
The total number leaf nodes of the tree is: 27
The split index is [19, 7, 20, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 8

After shuffle, the attributes is [12, 13, 14, 15, 20, 7, 21, 9, 19, 1, 17] and length is 11
The total number leaf nodes of the tree is: 28
The split index is [19, 7, 21, 1, 20, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 9

After shuffle, the attributes is [14, 1, 15, 13, 17, 19, 7, 9, 12, 20, 21] and length is 11
The total number leaf nodes of the tree is: 27
The split index is [19, 7, 20, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 10

After shuffle, the attributes is [7, 15, 14, 1, 21, 12, 13, 19, 17, 9, 20] and length is 11
The total number leaf nodes of the tree is: 26
The split index is [19, 7, 1, 21, 9, 20]
Number of children on the root 9
The tree depth is 6
Create TREE 11

After shuffle, the attributes is [20, 13, 12, 19, 9, 1, 17, 7, 15, 14, 21] and length is 11
The total number leaf nodes of the tree is: 23
The split index is [19, 7, 1, 21, 20, 17, 14]
Number of children on the root 9
The tree depth is 7
Create TREE 12

After shuffle, the attributes is [15, 21, 19, 20, 14, 7, 9, 12, 13, 17, 1] and length is 11
The total number leaf nodes of the tree is: 26
The split index is [19, 7, 1, 21, 9, 20]
Number of children on the root 9
The tree depth is 6
Create TREE 13

After shuffle, the attributes is [20, 19, 21, 1, 13, 9, 15, 14, 12, 7, 17] and length is 11
The total number leaf nodes of the tree is: 27
The split index is [19, 7, 20, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 14

After shuffle, the attributes is [13, 15, 12, 21, 9, 14, 17, 20, 7, 19, 1] and length is 11
The total number leaf nodes of the tree is: 26
The split index is [19, 7, 1, 21, 20, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 15

After shuffle, the attributes is [13, 14, 9, 21, 20, 1, 19, 17, 12, 15, 7] and length is 11
The total number leaf nodes of the tree is: 26
The split index is [19, 7, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 7
Create TREE 16

After shuffle, the attributes is [19, 13, 12, 14, 7, 15, 1, 17, 20, 9, 21] and length is 11
The total number leaf nodes of the tree is: 24
The split index is [19, 7, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 7
Create TREE 17

After shuffle, the attributes is [7, 14, 21, 15, 13, 17, 12, 1, 9, 19, 20] and length is 11
The total number leaf nodes of the tree is: 24
The split index is [19, 7, 1, 21, 9, 20]
Number of children on the root 9
The tree depth is 6
Create TREE 18

After shuffle, the attributes is [12, 20, 17, 1, 19, 21, 7, 15, 13, 14, 9] and length is 11
The total number leaf nodes of the tree is: 26
The split index is [19, 7, 1, 21, 20, 9, 14]
Number of children on the root 9
The tree depth is 7
Create TREE 19

After shuffle, the attributes is [14, 1, 19, 12, 15, 7, 17, 9, 20, 13, 21] and length is 11
The total number leaf nodes of the tree is: 27
The split index is [19, 7, 20, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Create TREE 20

After shuffle, the attributes is [21, 9, 14, 1, 20, 7, 13, 15, 19, 12, 17] and length is 11
The total number leaf nodes of the tree is: 27
The split index is [19, 7, 20, 1, 21, 9, 17, 14]
Number of children on the root 9
The tree depth is 8
Accuracy: 1.0
(py27) zipeis-MacBook-Pro:test_forest zipeiwei$ 

