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

        #random_seed = 3
        #self.HYPER = hyperparameters.HyperParameter(random_seed)
        #self.raw_data = data.RawData(self.HYPER)
        

    def tearDown(self):

        """ Runs after every test.
        """
        
        pass


    def test_encode_features(self):

        """ Tests if encoder weights are trained when training entire model during
            Active learning algorithm.
        """
        
        random_seed = 3
        HYPER = hyperparameters.HyperParameter(random_seed)
        raw_data = data.RawData(HYPER)
        
        HYPER.EPOCHS = 2
        HYPER.PROFILES_PER_YEAR = 100
        HYPER.POINTS_PER_PROFILE = 50
        
        for grey_scale in [True, False]:
        
            HYPER.GREY_SCALE = grey_scale
        
            for spatial_features in ['average', 'histogram']:
            
                HYPER.SPATIAL_FEATURES = spatial_features
                
                # re-create raw_data for updating paths to data
                raw_data = data.RawData(HYPER)

                
                
                ### 1. Data preprocessing ###
                
                raw_data = data.import_consumption_profiles(
                    HYPER, 
                    raw_data, 
                    plot=False
                )
                raw_data = data.import_building_images(
                    HYPER, 
                    raw_data, 
                    plot=False
                )
                raw_data = data.import_meteo_data(
                    HYPER, 
                    raw_data, 
                    plot=False
                )
                dataset, raw_data = data.create_feature_label_pairs(
                    HYPER, 
                    raw_data
                )
                dataset = data.encode_time_features(
                    HYPER, 
                    dataset, 
                    silent=True
                )
                dataset = data.normalize_features(
                    HYPER, 
                    raw_data, 
                    dataset, 
                    silent=True
                )
                (
                    training_data, 
                    validation_data, 
                    spatial_test_data, 
                    temporal_test_data, 
                    spatemp_test_data
                ) = data.split_train_val_test(
                    HYPER, 
                    raw_data, 
                    dataset
                )
                spatemp_test_data = data.standardize_features(
                    HYPER, 
                    raw_data, 
                    spatemp_test_data, 
                    training_data, 
                    silent=True
                )
                temporal_test_data = data.standardize_features(
                    HYPER, 
                    raw_data, 
                    temporal_test_data, 
                    training_data, 
                    silent=True
                )
                spatial_test_data = data.standardize_features(
                    HYPER, 
                    raw_data, 
                    spatial_test_data, 
                    training_data, 
                    silent=True
                )
                validation_data = data.standardize_features(
                    HYPER, 
                    raw_data, 
                    validation_data, 
                    training_data, 
                    silent=True
                )
                training_data = data.standardize_features(
                    HYPER, 
                    raw_data, 
                    training_data, 
                    training_data, 
                    silent=True
                )


                ### 2. Prediction model ###
                
                (
                    loss_object, 
                    optimizer, 
                    loss_function, 
                    mean_loss
                ) = prediction.initialize_optimizer(HYPER)
                models = prediction.build_prediction_model(
                    HYPER, 
                    raw_data, 
                    training_data, 
                    plot=False
                )
                _, _ = prediction.train_model(
                    HYPER, 
                    models.prediction_model, 
                    training_data, 
                    validation_data, 
                    raw_data,
                    loss_object, 
                    optimizer, 
                    mean_loss, 
                    silent=False
                )
                prediction.save_prediction_model(
                    HYPER, 
                    raw_data, 
                    models.prediction_model, 
                    'initial'
                )
                prediction.save_encoder_and_predictor_weights(
                    HYPER, 
                    raw_data, 
                    models
                )
                
                
                ### 3. Active learning ###

                dataset_list = [
                    spatial_test_data, 
                    temporal_test_data, 
                    spatemp_test_data
                ]

                for cand_subsample_act_lrn in [
                    None, 
                    5, 
                    100, 
                    100000013230
                ]:
                    
                    HYPER.CAND_SUBSAMPLE_ACT_LRN = cand_subsample_act_lrn

                    for pred_type in [
                        "spatial", 
                        "temporal", 
                        "spatio-temporal"
                    ]:
                        
                        # choose corresponding test data of currently iterated pred_type
                        if pred_type=='spatial':
                            dataset = dataset_list[0]
                            
                        if pred_type=='temporal':
                            dataset = dataset_list[1]
                            
                        if pred_type=='spatio-temporal':
                            dataset = dataset_list[2]
                            
                            
                        # get available index set
                        available_index_set_update = set(
                            np.arange(
                                dataset.n_datapoints
                            )
                        )
                            
                        # iterate over all sort variables that are chosen to be considered
                        for AL_variable in [
                            "X_t", 
                            "X_s1", 
                            "X_st", 
                            "X_(t,s)",
                            "Y_hat_(t,s)", 
                            "Y_(t,s)"
                        ]:

                            (
                                candidate_encoded, 
                                cand_sub_index 
                            ) = activelearning.encode_features(
                                HYPER,
                                raw_data,
                                models,
                                dataset,
                                available_index_set_update,
                                AL_variable,
                            )
                            
                            self.assertEqual(
                                len(candidate_encoded), 
                                len(cand_sub_index)
                            )
                            self.assertLessEqual(
                                max(cand_sub_index), 
                                len(dataset.Y)
                            )
                            
                            if AL_variable == 'X_t':
                                encoding_size = HYPER.ENCODING_NODES_X_t

                            elif AL_variable == 'X_s1':
                                encoding_size = HYPER.ENCODING_NODES_X_s
                                
                            elif AL_variable == 'X_st':
                                encoding_size = HYPER.ENCODING_NODES_X_st
                                
                            elif AL_variable == 'X_(t,s)':
                                encoding_size = HYPER.ENCODING_NODES_X_joint
                                
                            elif AL_variable == 'Y_hat_(t,s)':
                                encoding_size = HYPER.PREDICTION_WINDOW
                                
                            elif AL_variable == 'Y_(t,s)':
                                encoding_size = HYPER.PREDICTION_WINDOW
                                
                            self.assertEqual(
                                candidate_encoded.shape[1], 
                                encoding_size
                            )


    def test_compute_clusters(self):

        """ Tests if clusters are have labels 0 to n_cluster-1 and if 
            cluster centers are transformed accordingly.
        """ 
        
        random_seed = 3
        HYPER = hyperparameters.HyperParameter(random_seed)
        raw_data = data.RawData(HYPER)
        
        HYPER.EPOCHS = 2
        HYPER.PROFILES_PER_YEAR = 100
        HYPER.POINTS_PER_PROFILE = 50
 
            
        ### 1. Data preprocessing ###
        
        raw_data = data.import_consumption_profiles(
            HYPER, 
            raw_data, 
            plot=False
        )
        raw_data = data.import_building_images(
            HYPER, 
            raw_data, 
            plot=False
        )
        raw_data = data.import_meteo_data(
            HYPER, 
            raw_data, 
            plot=False
        )
        dataset, raw_data = data.create_feature_label_pairs(
            HYPER, 
            raw_data
        )
        dataset = data.encode_time_features(
            HYPER, 
            dataset, 
            silent=True
        )
        dataset = data.normalize_features(
            HYPER, 
            raw_data, 
            dataset, 
            silent=True
        )
        (
            training_data, 
            validation_data, 
            spatial_test_data, 
            temporal_test_data, 
            spatemp_test_data
        ) = data.split_train_val_test(
            HYPER, 
            raw_data, 
            dataset
        )
        spatemp_test_data = data.standardize_features(
            HYPER, 
            raw_data, 
            spatemp_test_data, 
            training_data, 
            silent=True
        )
        temporal_test_data = data.standardize_features(
            HYPER, 
            raw_data, 
            temporal_test_data, 
            training_data, 
            silent=True
        )
        spatial_test_data = data.standardize_features(
            HYPER, 
            raw_data, 
            spatial_test_data, 
            training_data, 
            silent=True
        )
        validation_data = data.standardize_features(
            HYPER, 
            raw_data, 
            validation_data, 
            training_data, 
            silent=True
        )
        training_data = data.standardize_features(
            HYPER, 
            raw_data, 
            training_data, 
            training_data, 
            silent=True
        )


        ### 2. Prediction model ###
        
        (
            loss_object, 
            optimizer, 
            loss_function, 
            mean_loss
        ) = prediction.initialize_optimizer(
            HYPER
        )
        models = prediction.build_prediction_model(
            HYPER, 
            raw_data, 
            training_data, 
            plot=False
        )
        _, _ = prediction.train_model(
            HYPER, 
            models.prediction_model, 
            training_data, 
            validation_data, 
            raw_data,
            loss_object, 
            optimizer, 
            mean_loss, 
            silent=False
        )
        prediction.save_prediction_model(
            HYPER, 
            raw_data, 
            models.prediction_model, 
            'initial'
        )
        prediction.save_encoder_and_predictor_weights(
            HYPER, 
            raw_data, 
            models
        )
        
        
        ### 3. Active learning ###

        dataset_list = [
            spatial_test_data, 
            temporal_test_data, 
            spatemp_test_data
        ]

        # iterate over possible candidate sub sample sizes
        for cand_subsample_act_lrn in [
            None, 
            5, 
            100, 
            100000013230
        ]:
            
            HYPER.CAND_SUBSAMPLE_ACT_LRN = cand_subsample_act_lrn
            
            # iterate over all prediction types
            for pred_type in [
                "spatial", 
                "temporal", 
                "spatio-temporal"
            ]:
                
                # choose corresponding test data of currently iterated pred_type
                if pred_type=='spatial':
                    dataset = dataset_list[0]
                    
                if pred_type=='temporal':
                    dataset = dataset_list[1]
                    
                if pred_type=='spatio-temporal':
                    dataset = dataset_list[2]
                    
                # get available index set
                available_index_set_update = set(
                    np.arange(
                        dataset.n_datapoints
                    )
                )
                    
                ### Set batch size ###

                # Compute total data budget
                data_budget = math.floor(
                    HYPER.DATA_BUDGET_ACT_LRN * dataset.n_datapoints
                )

                # compute the batch siz of this iteration
                cand_batch_size = HYPER.CAND_BATCH_SIZE_ACT_LRN * data_budget

                # if exceeding candidate data subsample, adjust batch size
                if HYPER.CAND_SUBSAMPLE_ACT_LRN is not None:

                    cand_batch_size = min(
                        cand_batch_size, 
                        HYPER.CAND_SUBSAMPLE_ACT_LRN
                    )

                # transform cand_batch_siz to integer
                cand_batch_size = int(cand_batch_size)
                    
                # iterate over all sort variables that are chosen to be considered
                for AL_variable in [
                    "X_t", 
                    "X_s1", 
                    "X_st", 
                    "X_(t,s)",
                    "Y_hat_(t,s)", 
                    "Y_(t,s)"
                ]:


                    ### Encode data points ###

                    (
                        candidate_encoded, 
                        cand_sub_index 
                    ) = activelearning.encode_features(
                        HYPER,
                        raw_data,
                        models,
                        dataset,
                        available_index_set_update,
                        AL_variable
                    )
                    
                    
                    ### Calculate clusters ###

                    (
                        cand_labels, 
                        cand_centers, 
                        n_clusters
                    ) = activelearning.compute_clusters(
                        HYPER, 
                        candidate_encoded, 
                        cand_batch_size
                    )

                    self.assertEqual(n_clusters, max(cand_labels)+1)
                    self.assertEqual(n_clusters, len(cand_centers))
                    self.assertEqual(0, min(cand_labels))
    

    def test_compute_similarity(self):

        """ Tests if similarities are calculated with higher values for closer
            points given each of the possible metrics.
        """ 
        
        random_seed = 3
        HYPER = hyperparameters.HyperParameter(random_seed)
        raw_data = data.RawData(HYPER)
        
        n_dims = 500
        
        # create some data manually
        cluster_centers = [np.ones((n_dims,)) * 4.5]
        p1 = np.ones((n_dims,)) * 2
        p2 = np.ones((n_dims,)) * 6
        
        cluster_labels = [0, 0]
        encoding = [p1, p2]
        
        for metric in [rbf_kernel, laplacian_kernel]:
        
            HYPER.METRIC_DISTANCES = [metric]
        
            similarity_array = activelearning.compute_similarity(
                HYPER, encoding, cluster_labels, cluster_centers, silent=True
            )
            
            self.assertLessEqual(similarity_array[0], similarity_array[1])
            self.assertEqual(similarity_array.shape, (len(encoding),))

        
    def test_encoder_training(self):

        """ Tests if encoder weights are trained when training entire model 
        during Active learning algorithm.
        """
            
        random_seed = 3
        HYPER = hyperparameters.HyperParameter(random_seed)
        raw_data = data.RawData(HYPER)
        
        HYPER.EPOCHS = 2
        HYPER.PROFILES_PER_YEAR = 100
        HYPER.POINTS_PER_PROFILE = 50
        HYPER.SAVE_ACT_LRN_MODELS = True
        HYPER.SPATIAL_FEATURES = "histogram"
        

        ### 1. Data preprocessing ###
        
        raw_data = data.import_consumption_profiles(
            HYPER, 
            raw_data, 
            plot=False
        )
        raw_data = data.import_building_images(
            HYPER, 
            raw_data, 
            plot=False
        )
        raw_data = data.import_meteo_data(
            HYPER, 
            raw_data, 
            plot=False
        )
        dataset, raw_data = data.create_feature_label_pairs(
            HYPER, 
            raw_data
        )
        dataset = data.encode_time_features(
            HYPER, 
            dataset, 
            silent=True
        )
        dataset = data.normalize_features(
            HYPER, 
            raw_data, 
            dataset, 
            silent=True
        )
        (
            training_data, 
            validation_data, 
            spatial_test_data, 
            temporal_test_data, 
            spatemp_test_data
        ) = data.split_train_val_test(
            HYPER, 
            raw_data, 
            dataset
        )
        spatemp_test_data = data.standardize_features(
            HYPER, 
            raw_data, 
            spatemp_test_data, 
            training_data, 
            silent=True
        )
        temporal_test_data = data.standardize_features(
            HYPER, 
            raw_data, 
            temporal_test_data, 
            training_data, 
            silent=True
        )
        spatial_test_data = data.standardize_features(
            HYPER, 
            raw_data, 
            spatial_test_data, 
            training_data, 
            silent=True
        )
        validation_data = data.standardize_features(
            HYPER, 
            raw_data, 
            validation_data, 
            training_data, 
            silent=True
        )
        training_data = data.standardize_features(
            HYPER, 
            raw_data, 
            training_data, 
            training_data, 
            silent=True
        )


        ### 2. Prediction model ###
        
        (
            loss_object, 
            optimizer, 
            loss_function, 
            mean_loss
        ) = prediction.initialize_optimizer(
            HYPER
        )
        models = prediction.build_prediction_model(
            HYPER, 
            raw_data, 
            training_data, 
            plot=False
        )
        _, _ = prediction.train_model(
            HYPER, 
            models.prediction_model, 
            training_data, 
            validation_data, 
            raw_data,
            loss_object, 
            optimizer, 
            mean_loss, 
            silent=False
        )
        prediction.save_prediction_model(
            HYPER, 
            raw_data, 
            models.prediction_model, 
            'initial'
        )
        prediction.save_encoder_and_predictor_weights(
            HYPER, 
            raw_data, 
            models
        )
        
        
        ### 3. Active learning ###

        dataset_list = [
            spatial_test_data, 
            temporal_test_data, 
            spatemp_test_data
        ]

        for pred_type in ["spatio-temporal"]:
            
            # choose corresponding test data of currently iterated pred_type
            if pred_type=='spatial':
                dataset = dataset_list[0]
                
            if pred_type=='temporal':
                dataset = dataset_list[1]
                
            if pred_type=='spatio-temporal':
                dataset = dataset_list[2]
                
            # iterate over all possible AL variables
            for AL_variable in ["X_t", "X_(t,s)", "Y_(t,s)"]:
            
                # iterate over all possible AL variants
                for method in ["cluster-rnd", "cluster-avg"]:

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
                    
                    # test AL with currently iterated AL variable and variant
                    result =  activelearning.feature_embedding_AL(
                        HYPER, 
                        pred_type, 
                        models, 
                        raw_data, 
                        training_data, 
                        dataset,
                        loss_object, 
                        optimizer, 
                        mean_loss, 
                        loss_function,
                        method=method, 
                        AL_variable=AL_variable, 
                        silent=False
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
                    # CAUTION: these two tests will fail if first layer of predictor is X_s or X_st instead
                    # of X_t
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
        
