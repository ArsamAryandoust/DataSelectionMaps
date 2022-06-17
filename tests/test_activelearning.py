import sys
sys.path.insert(0, '../src')

import unittest
import activelearning

import hyperparameters
import data
import prediction

import numpy as np
import math

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import cosine_similarity

class TestActiveLearning(unittest.TestCase):

    """ Tests functions defined in data.py
    """


    @classmethod
    def setUpClass(cls):
        
        """ Runs once before the first test.
        """

        pass


    @classmethod
    def tearDownClass(cls):
        
        """ Runs once after the last test.
        """

        pass


    def setUp(self):
        
        """ Runs before every test.
        """


    def tearDown(self):

        """ Runs after every test.
        """
        
        pass


    def test_encode_features(self):

        """ Tests if encoder weights are trained when training entire model during
            Active learning algorithm.
        """
        
        
        

 
                    


# write this to use 'python test_data.py' instead of 'python -m unittest test_data.py'
if __name__ == '__main__':
   
    unittest.main()
        
