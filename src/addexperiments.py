import math
import random

import numpy as np
import tensorflow as tf

from data import Dataset
from prediction import train_model, test_model
from prediction import load_encoder_and_predictor_weights

import activelearning

def test_sequence_importance_AL(
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
    method,
    AL_variable,
    silent=True
):

    AL_result = activelearning.feature_embedding_AL(
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
        method, 
        AL_variable 
    )
    
    AL_result = randomize_sequence_AL(
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
        AL_result,
        method,
        AL_variable
    )
    
    return AL_result
    

def randomize_sequence_AL(
    HYPER,
    pred_type,
    models,
    raw_data,
    train_data,
    candidate_dataset,
    loss_object,
    optimizer,
    mean_loss,
    loss_function,
    AL_result,
    method,
    AL_variable=None,
    silent=True
):
    
    """ Tests the importance of the query sequence for passed AL results """

    if HYPER.TEST_SEQUENCE_IMPORTANCE:
        if not silent:
            # create a progress bar for training
            progbar_seqimportance = tf.keras.utils.Progbar(HYPER.N_ITER_ACT_LRN)

            # tell us what we are doing
            print('Testing sequence importance for')
            
            print(
                'prediction type:                      {}'.format(
                    pred_type
                )
            )
            
            print(
                'query variable:                       {}'.format(
                    AL_variable
                )
            )
            
            print(
                'query variant:                        {}'.format(
                    method
                )
            )
        
        
        ### Load model weights ###

        # Note: if you load entire initial models, instead of their weights only,
        # network configuration information is lost and tf will not train encoders
        # alongside training the main prediction model.
        models = load_encoder_and_predictor_weights(
            raw_data, 
            models, 
            pred_type
        )


        ### Start AL algorithm with random sequence selection ###

        # initialize some values
        data_budget_total = math.floor(
            HYPER.DATA_BUDGET_ACT_LRN * candidate_dataset.n_datapoints
        )
        # compute data budget available in each query iteration
        data_budget_per_iter = math.floor(
            data_budget_total / HYPER.N_ITER_ACT_LRN
        )
        picked_cand_index_set = set()
        available_index_set_update = AL_result['picked_cand_index_set']
        data_counter = 0
        
        # start AL iterations
        for iteration in range(HYPER.N_ITER_ACT_LRN):

            ### Choose training batch ###

            # Create a random splitting array
            batch_index_list = random.sample(
                available_index_set_update, 
                data_budget_per_iter
            )
            
            # update candidate indices and data counter
            picked_cand_index_set = picked_cand_index_set.union(
                set(batch_index_list)
            )
            picked_cand_index_list = list(picked_cand_index_set)
            data_counter = len(picked_cand_index_list)


            ### Create training data ####
            
            if HYPER.EXTEND_TRAIN_DATA_ACT_LRN:
                # get share of training data from pool of possible testing points
                X_t_ord_1D_new_train = np.concatenate(
                    (
                        train_data.X_t_ord_1D, 
                        candidate_dataset.X_t_ord_1D[picked_cand_index_list]
                    ), 
                    axis=0
                )
                X_t_new_train = np.concatenate(
                    (
                        train_data.X_t, 
                        candidate_dataset.X_t[picked_cand_index_list]
                    ), 
                    axis=0
                )
                X_s_new_train = np.concatenate(
                    (
                        train_data.X_s, 
                        candidate_dataset.X_s[picked_cand_index_list]
                    ), 
                    axis=0
                )
                X_st_new_train = np.concatenate(
                    (
                        train_data.X_st, 
                        candidate_dataset.X_st[picked_cand_index_list]
                    ), 
                    axis=0
                )
                Y_new_train = np.concatenate(
                    (
                        train_data.Y, 
                        candidate_dataset.Y[picked_cand_index_list]
                    ), 
                    axis=0
                )

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_train = np.concatenate(
                        (
                            train_data.X_s1, 
                            candidate_dataset.X_s1[picked_cand_index_list]
                        ), 
                        axis=0
                    )
                else:
                    X_s1_new_train = 0
            
            else:
                # sort all initial candidate data features with the same array
                X_t_ord_1D_new_train = candidate_dataset.X_t_ord_1D[batch_index_list]
                X_t_new_train = candidate_dataset.X_t[batch_index_list]
                X_s_new_train = candidate_dataset.X_s[batch_index_list]
                X_st_new_train = candidate_dataset.X_st[batch_index_list]
                Y_new_train = candidate_dataset.Y[batch_index_list]

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_train = candidate_dataset.X_s1[batch_index_list]
                else:
                    X_s1_new_train = 0
                    

            ### Update picked_cand_index_list ###

            # update candidate data if chosen so
            if HYPER.RED_CAND_DATA_ACT_LRN:
                # update set of available indices
                available_index_set_update = (
                    available_index_set_update - picked_cand_index_set
                )


            ### Create (updated) validation data ###
            
            if HYPER.UPD_VAL_DATA_ACT_LRN:
                # create new validation data by deleting the batch
                X_t_ord_1D_new_val = np.delete(
                    candidate_dataset.X_t_ord_1D, picked_cand_index_list, 0
                )
                X_t_new_val = np.delete(
                    candidate_dataset.X_t, picked_cand_index_list, 0
                )
                X_s_new_val = np.delete(
                    candidate_dataset.X_s, picked_cand_index_list, 0
                )
                X_st_new_val = np.delete(
                    candidate_dataset.X_st, picked_cand_index_list, 0
                )
                Y_new_val = np.delete(
                    candidate_dataset.Y, picked_cand_index_list, 0
                )

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_val = np.delete(
                        candidate_dataset.X_s1, picked_cand_index_list, 0
                    )
                else:
                    X_s1_new_val = 0

            else:
                # create new validation data by copying initial candidates
                X_t_ord_1D_new_val = candidate_dataset.X_t_ord_1D
                X_t_new_val = candidate_dataset.X_t
                X_s_new_val = candidate_dataset.X_s
                X_st_new_val = candidate_dataset.X_st
                Y_new_val = candidate_dataset.Y

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_val = candidate_dataset.X_s1
                else:
                    X_s1_new_val = 0
                    
                    
            ### Train with new batch ###

            # bundle chosen batch of candidate data points as Dataset object
            new_train_batch = Dataset(
                X_t_ord_1D_new_train,
                X_t_new_train, 
                X_s_new_train, 
                X_s1_new_train, 
                X_st_new_train, 
                Y_new_train
            )

            # bundle updated data points as Dataset object for validation. This 
            # avoids unwanted shuffling
            new_val_data = Dataset(
                X_t_ord_1D_new_val,
                X_t_new_val, 
                X_s_new_val, 
                X_s1_new_val, 
                X_st_new_val, 
                Y_new_val
            )

            # train model with new data
            train_hist_batch, val_hist_batch = train_model(
                HYPER,
                models.prediction_model,
                new_train_batch,
                new_val_data,
                raw_data,
                loss_object,
                optimizer,
                mean_loss
            )

            # keep track of loss histories
            if iteration == 0:
                train_hist = train_hist_batch
                val_hist = val_hist_batch
            else:
                train_hist = np.concatenate((train_hist, train_hist_batch))
                val_hist = np.concatenate((val_hist, val_hist_batch))

            if not silent:
                # increment progress bar
                progbar_seqimportance.add(1)
                
                
        ### Create test dataset and predict ###

        # create new validation data by deleting the batch of picked data from 
        # candidate dataset
        X_t_ord_1D_test = np.delete(
            candidate_dataset.X_t_ord_1D, 
            picked_cand_index_list, 
            0
        )
        
        X_t_test = np.delete(
            candidate_dataset.X_t, 
            picked_cand_index_list, 
            0
        )
        
        X_s_test = np.delete(
            candidate_dataset.X_s, 
            picked_cand_index_list, 
            0
        )
        
        X_st_test = np.delete(
            candidate_dataset.X_st, 
            picked_cand_index_list, 
            0
        )
        
        Y_test = np.delete(
            candidate_dataset.Y, 
            picked_cand_index_list, 
            0
        )
        
        if HYPER.SPATIAL_FEATURES != "image":
            X_s1_test = np.delete(
                candidate_dataset.X_s1, 
                picked_cand_index_list, 
                0
            )
        else:
            X_s1_test = 0
            
        # create a copy of candidate test data
        test_data = Dataset(
            X_t_ord_1D_test,
            X_t_test, 
            X_s_test, 
            X_s1_test, 
            X_st_test, 
            Y_test
        )
        
        # Predict on candidate datapoints that are not in training data
        title = '{} {} {}'.format(pred_type, AL_variable, method)
        
        test_loss = test_model(
            HYPER, 
            title, 
            models.prediction_model, 
            test_data, 
            raw_data, 
            mean_loss,
            loss_function 
        )
        
        random_sequence = {
            'train_hist': train_hist,
            'val_hist': val_hist,
            'test_loss': test_loss
        }
        AL_sequence = {
            'train_hist': AL_result['train_hist'],
            'val_hist': AL_result['val_hist'],
            'test_loss': AL_result['test_loss']
        }
        result_dict = {
            'random_sequence': random_sequence,
            'AL_sequence' : AL_sequence
        }

        if not silent: 
            # Indicate termination of execute
            print('---' * 20)

    return result_dict


def test_subsample_importance_AL(
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
    method,
    AL_variable,
    silent=True
):
    
    """ Tests heuristic methods for passed AL results """

    if not silent:
        print('Testing subsample importance')
        print('prediction type:                      {}'.format(pred_type))
        print('query variable:                       {}'.format(AL_variable))
        print('query variant:                        {}'.format(method))
        progbar = tf.keras.utils.Progbar(len(HYPER.CAND_SUBSAMPLE_TEST_LIST))
            
    # save original hyper parameter values
    original_subsample = HYPER.CAND_SUBSAMPLE_ACT_LRN
            
    heuristic_results_list = []
    for heuristic_value in HYPER.CAND_SUBSAMPLE_TEST_LIST:
        HYPER.CAND_SUBSAMPLE_ACT_LRN = heuristic_value
        
        results = activelearning.feature_embedding_AL(
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
            method, 
            AL_variable=AL_variable, 
        )
        
        heuristic_results_list.append(results)
        if not silent:
            progbar.add(1)
    
    HYPER.CAND_SUBSAMPLE_ACT_LRN = original_subsample
    result_dict = {
        'heuristics_subsample' : heuristic_results_list
    }

    return result_dict

def test_pointspercluster_importance_AL(
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
    method,
    AL_variable,
    silent=True
):
    
    """ Tests heuristic methods for passed AL results """

    if not silent:
        print('Testing points per cluster importance')
        print('prediction type:                      {}'.format(pred_type))
        print('query variable:                       {}'.format(AL_variable))
        print('query variant:                        {}'.format(method))
        progbar = tf.keras.utils.Progbar(len(HYPER.POINTS_PERCLUSTER_TEST_LIST))
            
    # save original hyper parameter values
    original_pointspercluster = HYPER.POINTS_PER_CLUSTER_ACT_LRN
            
    heuristic_results_list = []
    for heuristic_value in HYPER.CAND_SUBSAMPLE_TEST_LIST:
        HYPER.CAND_SUBSAMPLE_ACT_LRN = heuristic_value
        
        results = activelearning.feature_embedding_AL(
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
            method, 
            AL_variable=AL_variable, 
        )
        
        heuristic_results_list.append(results)
        if not silent:
            progbar.add(1)
    
    HYPER.POINTS_PER_CLUSTER_ACT_LRN = original_pointspercluster
    result_dict = {
        'heuristics_pointspercluster' : heuristic_results_list
    }

    return result_dict
    
def test_querybycoordinate_importance_AL(
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
    method,
    AL_variable,
    silent=True
):
    
    """ Tests heuristic methods for passed AL results """

    if not silent:
        print('Testing query by coordinate importance')
        print('prediction type:                      {}'.format(pred_type))
        print('query variable:                       {}'.format(AL_variable))
        print('query variant:                        {}'.format(method))
        progbar = tf.keras.utils.Progbar(len(HYPER.CAND_SUBSAMPLE_TEST_LIST))
            
    # save original hyper parameter values
    original_subsample = HYPER.CAND_SUBSAMPLE_ACT_LRN
            
    heuristic_results_list = []
    for heuristic_value in HYPER.CAND_SUBSAMPLE_TEST_LIST:
        HYPER.CAND_SUBSAMPLE_ACT_LRN = heuristic_value
        
        results = activelearning.feature_embedding_AL(
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
            method, 
            AL_variable=AL_variable, 
        )
        
        heuristic_results_list.append(results)
        if not silent:
            progbar.add(1)
    
    HYPER.CAND_SUBSAMPLE_ACT_LRN = original_subsample
    result_dict = {
        'heuristics_querybycoordinate' : heuristic_results_list
    }

    return result_dict
    
    return heuristic_results_dict
