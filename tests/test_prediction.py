import sys
sys.path.insert(0, '../src')

import unittest
import prediction

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

        random_seed = 3
        self.HYPER = hyperparameters.HyperParameter(random_seed)
        self.raw_data = data.RawData(self.HYPER)

        # call functions for data preparation
        self.raw_data = data.import_consumption_profiles(
            self.HYPER, 
            self.raw_data, 
            plot=False
        )
        self.raw_data = data.import_building_images(
            self.HYPER, 
            self.raw_data, 
            plot=False
        )
        self.raw_data = data.import_meteo_data(
            self.HYPER, 
            self.raw_data, 
            plot=False
        )
        dataset, self.raw_data = data.create_feature_label_pairs(
            self.HYPER, 
            self.raw_data
        )
        dataset = data.encode_time_features(
            self.HYPER, 
            dataset
        )
        dataset = data.normalize_features(
            self.HYPER, 
            self.raw_data, 
            dataset
        )
        (
            training_data, 
            validation_data, 
            spatial_test_data, 
            temporal_test_data, 
            spatemp_test_data
        ) = data.split_train_val_test(self.HYPER, self.raw_data, dataset)
        self.spatemp_test_data = data.standardize_features(
            self.HYPER, 
            self.raw_data, 
            spatemp_test_data, 
            training_data
        )
        self.temporal_test_data = data.standardize_features(
            self.HYPER, 
            self.raw_data, 
            temporal_test_data, 
            training_data
        )
        self.spatial_test_data = data.standardize_features(
            self.HYPER, 
            self.raw_data, 
            spatial_test_data, 
            training_data
        )
        self.validation_data = data.standardize_features(
            self.HYPER, 
            self.raw_data, 
            validation_data, 
            training_data
        )
        self.training_data = data.standardize_features(
            self.HYPER, 
            self.raw_data, 
            training_data, 
            training_data
        )

        (
            self.loss_object, 
            self.optimizer, 
            self.loss_function, 
            self.mean_loss
        ) = prediction.initialize_optimizer(
            self.HYPER
        )
        


    def tearDown(self):

        """ Runs after every test.
        """
        
        pass

	
    def test_encoder_training(self):

        """ Tests if encoder weights are trained when training entire model.
        """
        
        # build prediction models
        models = prediction.build_prediction_model(
            self.HYPER, 
            self.raw_data, 
            self.training_data, 
            plot=False
        )
        
        # encoder weights before training model
        first_layer_encoder_before = (
            models.X_t_encoder.trainable_variables[0].numpy()
        )
        last_layer_encoder_before = (
            models.X_t_encoder.trainable_variables[-1].numpy()
        )
        
        # prediction model weights before training model
        first_layer_predictor_before = (
            models.prediction_model.trainable_variables[0].numpy()
        )
        last_layer_predictor_before = (
            models.prediction_model.trainable_variables[-1].numpy()
        )
        
        # model training
        _, _ = prediction.train_model(
            self.HYPER, 
            models.prediction_model, 
            self.training_data, 
            self.validation_data, 
            self.raw_data,
            self.loss_object, 
            self.optimizer, 
            self.mean_loss,
            silent=False, 
            plot=False
        )
        
        # encoder weights after training model
        first_layer_encoder_after = (
            models.X_t_encoder.trainable_variables[0].numpy()
        )
        last_layer_encoder_after = (
            models.X_t_encoder.trainable_variables[-1].numpy()
        )
        
        # prediction model weights after training model
        first_layer_predictor_after = (
            models.prediction_model.trainable_variables[0].numpy()
        )
        last_layer_predictor_after = (
            models.prediction_model.trainable_variables[-1].numpy()
        )
        
        # test if last layers are different before and after training
        self.assertFalse(
            np.array_equal(
                last_layer_encoder_before, 
                last_layer_encoder_after
            )
        )
        self.assertFalse(
            np.array_equal(
                last_layer_predictor_before, 
                last_layer_predictor_after
            )
        )
        
        # test if first layers are different before and after training
        self.assertFalse(
            np.array_equal(
                first_layer_encoder_before, 
                first_layer_encoder_after
            )
        )
        self.assertFalse(
            np.array_equal(
                first_layer_predictor_before, 
                first_layer_predictor_after
            )
        )
        
        # test if first layers of encoer and predictor are equal before and after training
        self.assertTrue(
            np.array_equal(
                first_layer_encoder_before, 
                first_layer_predictor_before
            )
        )
        self.assertTrue(
            np.array_equal(
                first_layer_encoder_after, 
                first_layer_predictor_after
            )
        )
        
        
        
# write this to use 'python test_data.py' instead of 'python -m unittest test_data.py'
if __name__ == '__main__':
   
    unittest.main()
        
