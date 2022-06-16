import sys
sys.path.insert(0, '../src')

import unittest
import vis_results

import data
import hyperparameters
import numpy as np

class TestPrediction(unittest.TestCase):

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

        self.HYPER_VIS = vis_results.HyperParameterVisualizing()
        


    def tearDown(self):

        """ Runs after every test.
        """
        
        pass

	
    def test_test_hyper(self):

        """ Tests if hyper paramters match.
        """
        vis_results.test_hyper(self.HYPER_VIS)
        
    def test_plot_train_val_hist(self):

        """
        """
        vis_results.plot_train_val_hist(self.HYPER_VIS) 
        
    def test_plot_subsampling_heuristics(self):

        """
        """
        vis_results.plot_subsampling_heuristics(self.HYPER_VIS)  
    
    def test_plot_pointspercluster_heuristics(self):

        """
        """
        vis_results.plot_pointspercluster_heuristics(self.HYPER_VIS)  
        
    def test_plot_querybycoordinate_heuristics(self):

        """
        """
        vis_results.plot_querybycoordinate_heuristics(self.HYPER_VIS) 
        
    def test_plot_sequence_importance(self):

        """
        """
        vis_results.plot_sequence_importance(self.HYPER_VIS) 
        
    def test_plot_results_summary(self):

        """
        """
        vis_results.plot_results_summary(self.HYPER_VIS) 
        vis_results.plot_results_summary(self.HYPER_VIS, 'heuristic_summary') 
        
# write this to use 'python test_data.py' instead of 'python -m unittest test_data.py'
if __name__ == '__main__':
   
    unittest.main()
        
