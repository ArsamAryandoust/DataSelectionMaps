import os
import numpy as np
import pandas as pd


def saveallresults(
    HYPER,
    raw_data,
    RF_results, 
    AL_result_list, 
    PL_result_list

):

    """ Calls all other functions to save generated results """
    
    # save active learning results
    save_act_lrn_results(
        HYPER, 
        raw_data, 
        RF_results, 
        AL_result_list, 
        PL_result_list
    )

    # save hyper parameters
    save_hyper_params(
        HYPER, 
        raw_data
    )

    # save the prediction models
    save_act_lrn_models(
        HYPER, 
        raw_data, 
        AL_result_list, 
        PL_result_list
    )

    # save the test data sample
    save_act_lrn_test_sample(
        HYPER, 
        raw_data, 
        AL_result_list, 
        PL_result_list
    )

def save_act_lrn_models(
    HYPER, 
    raw_data, 
    AL_result_list, 
    PL_result_list
):

    """ Saves the actively trained prediction models. """

    if HYPER.SAVE_ACT_LRN_MODELS:

        for pred_type in HYPER.PRED_LIST_ACT_LRN:

            # get method_result_list of currently iterated prediction type
            var_result_list = AL_result_list[pred_type]

            # get random results
            PL_results = PL_result_list[pred_type]

            prediction_model = PL_results['prediction_model']

            # create the full path for saving random  prediction model
            saving_path = raw_data.path_to_AL_models + pred_type + '/'
        
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)
                
            path_to_model = saving_path + 'PL.h5'

            # save currently iterated model
            prediction_model.save(path_to_model)

            for AL_variable in HYPER.QUERY_VARIABLES_ACT_LRN:

                # get variable result list
                method_result_list = var_result_list[AL_variable]

                for method in HYPER.QUERY_VARIANTS_ACT_LRN:

                    # get result object and prediction model
                    AL_result = method_result_list[method]
                    prediction_model = AL_result['prediction_model']

                    # create the full path for saving currently iterated model
                    path_to_model = (
                        saving_path 
                        + AL_variable 
                        + ' '
                        + method 
                        + '.h5'
                    )

                    # save currently iterated model
                    prediction_model.save(path_to_model)


def save_act_lrn_results(
    HYPER, 
    raw_data, 
    RF_results, 
    AL_result_list, 
    PL_result_list
):

    """ Saves the active learning results, including number of iterations used, 
    time used for each iteration, share of data budget used, share of sensor 
    budget used, share of stream time budget used, testing loss baseline loss 
    and passive learning benchmark histories, validation histories and training 
    histories.
    """

    if HYPER.SAVE_ACT_LRN_RESULTS:

        for pred_type in HYPER.PRED_LIST_ACT_LRN:
            saving_path = raw_data.path_to_AL_results + pred_type + '/'
            
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)
                
            path_to_results_file = saving_path + 'results.csv'
            
            # create empty DataFrame
            result_df = pd.DataFrame()
            df_list = []

            # baseline results
            RF_loss = RF_results[pred_type]

            # get method_result_list of currently iterated prediction type
            var_result_list = AL_result_list[pred_type]

            ### Save PL results ###
            
            # get PL results
            PL_results = PL_result_list[pred_type]

            n_iterations = HYPER.N_ITER_ACT_LRN
            t_iterations = PL_results['iter_time']
            budget_usage = PL_results['budget_usage_hist'][-1]
            sensor_usage = PL_results['sensor_usage_hist'][-1]
            streamtime_usage = PL_results['streamtime_usage_hist'][-1]
            test_loss = PL_results['test_loss']

            train_hist = PL_results['train_hist']
            val_hist = PL_results['val_hist']

            col_name_train = '{} {} {} train'.format(pred_type, None, 'PL')
            col_name_val = '{} {} {} val'.format(pred_type, None, 'PL')

            meta_entry = np.array(
                [
                    n_iterations,
                    t_iterations,
                    budget_usage,
                    sensor_usage,
                    streamtime_usage,
                    RF_loss,
                    test_loss,
                ]
            )
            entry_train = np.concatenate((meta_entry, train_hist))
            entry_val = np.concatenate((meta_entry, val_hist))

            df_list.append(
                pd.DataFrame({col_name_train: pd.Series(entry_train)})
            )
            df_list.append(
                pd.DataFrame({col_name_val: pd.Series(entry_val)})
            )
            
            ### Prepare sequence imporatance for AL ### 
            if HYPER.TEST_SEQUENCE_IMPORTANCE:
                seqimportance_df = pd.DataFrame()
                df_list_seqimportance = []
                path_to_seqimportance_file = (
                    saving_path 
                    + 'sequence_importance.csv'
                )

            ### Prepare budget vs accuracy for PL ###
            
            path_to_budgetvsaccuracy_file = saving_path + 'budget_vs_accuracy.csv'
            budgetvsaccuracy_df = pd.DataFrame()
            budgetvsaccuracy_df_list = []
            data = np.rint(100 * np.array(PL_results['budget_usage_hist'])).astype(int)
            sensors = np.rint(100 * np.array(PL_results['sensor_usage_hist'])).astype(int)
            streamtimes = np.rint(100 * np.array(PL_results['streamtime_usage_hist'])).astype(int)
            val_loss_hist = PL_results['val_loss_hist']
            accuracy = np.rint(100 * (1 - np.minimum(1, val_loss_hist / RF_loss))).astype(int)
            
            col_name_data = '{} {} {} data'.format(
                pred_type, 
                None, 
                'PL'
            )
            col_name_sensors = '{} {} {} sensors'.format(
                pred_type, 
                None, 
                'PL'
            )
            col_name_streamtimes = '{} {} {} streamtimes'.format(
                pred_type, 
                None, 
                'PL'
            )
            col_name_accuracy = '{} {} {} accuracy'.format(
                pred_type, 
                None, 
                'PL'
            )

 
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_data: pd.Series(data)})
            )
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_sensors: pd.Series(sensors)})
            )
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_streamtimes: pd.Series(streamtimes)})
            )
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_accuracy: pd.Series(accuracy)})
            )
                
            ### Prepare picked times and spaces for PL ###
            path_to_spacetime_file = saving_path + 'spacetime_selection.csv'
            spacetime_df = pd.DataFrame()
            spacetime_df_list = []
            picked_times_index_hist = PL_results['picked_times_index_hist']
            picked_spaces_index_hist = PL_results['picked_spaces_index_hist']
            initial_sensors_list = PL_results['initial_sensors_list']
            
            col_name_initial_sensors = '{} - initial sensors'.format(
                pred_type
            )
            
            spacetime_df_list.append(
                  pd.DataFrame({col_name_initial_sensors: pd.Series(initial_sensors_list)})
              )
            
            for iteration in range(n_iterations):
                picked_times_list = picked_times_index_hist[iteration]
                picked_spaces_list = picked_spaces_index_hist[iteration]
                
                col_name_times = '{} {} {} - iter {} time'.format(
                    pred_type, 
                    None, 
                    'PL',
                    iteration
                )
                col_name_spaces = '{} {} {} - iter {} space'.format(
                    pred_type, 
                    None, 
                    'PL',
                    iteration
                )
                
                spacetime_df_list.append(
                    pd.DataFrame({col_name_times: pd.Series(picked_times_list)})
                )
                spacetime_df_list.append(
                    pd.DataFrame({col_name_spaces: pd.Series(picked_spaces_list)})
                )
            
            
            for AL_variable in HYPER.QUERY_VARIABLES_ACT_LRN:

                ### Save main AL results ### 
                method_result_list = var_result_list[AL_variable]

                for method in HYPER.QUERY_VARIANTS_ACT_LRN:

                    AL_result = method_result_list[method]

                    n_iterations = HYPER.N_ITER_ACT_LRN
                    t_iterations = AL_result['iter_time']
                    budget_usage = AL_result['budget_usage_hist'][-1]
                    sensor_usage = AL_result['sensor_usage_hist'][-1]
                    streamtime_usage = AL_result['streamtime_usage_hist'][-1]
                    test_loss = AL_result['test_loss']
                    delta_loss_RF = AL_result['test_loss'] - RF_loss
                    delta_loss_PL = (
                        AL_result['test_loss']
                        - PL_results['test_loss']
                    )

                    train_hist = AL_result['train_hist']
                    val_hist = AL_result['val_hist']
                    
                    col_name_train = '{} {} {} train'.format(
                        pred_type, 
                        AL_variable, 
                        method
                    )
                    col_name_val = '{} {} {} val'.format(
                        pred_type, 
                        AL_variable, 
                        method
                    )

                    meta_entry = np.array(
                        [
                            n_iterations,
                            t_iterations,
                            budget_usage,
                            sensor_usage,
                            streamtime_usage,
                            RF_loss,
                            test_loss,
                            delta_loss_RF,
                            delta_loss_PL,
                        ]
                    )
                    entry_train = np.concatenate((meta_entry, train_hist))
                    entry_val = np.concatenate((meta_entry, val_hist))

                    df_list.append(
                        pd.DataFrame({col_name_train: pd.Series(entry_train)})
                    )
                    df_list.append(
                        pd.DataFrame({col_name_val: pd.Series(entry_val)})
                    )
                    
                    ### Save sequence importance for AL ### 
                    if HYPER.TEST_SEQUENCE_IMPORTANCE:
                        train_loss_seqimportance = (
                            AL_result['seqimportance_train_loss']
                        )
                        val_loss_seqimportance = (
                            AL_result['seqimportance_val_loss']
                        )
                        test_loss_seqimportance = (
                            AL_result['seqimportance_test_loss']
                        )
                        meta_entry = np.array(
                            [
                                test_loss_seqimportance
                            ]
                        )
                        
                        entry_train_seqimportance = np.concatenate(
                            (
                                meta_entry, 
                                train_loss_seqimportance
                            )
                        )
                        entry_val_seqimportance = np.concatenate(
                            (
                                meta_entry, 
                                val_loss_seqimportance
                            )
                        )
                        
                        df_list_seqimportance.append(
                            pd.DataFrame(
                                {col_name_train: pd.Series(
                                    entry_train_seqimportance
                                )}
                            )
                        )
                        df_list_seqimportance.append(
                            pd.DataFrame(
                                {col_name_val: pd.Series(
                                    entry_val_seqimportance
                                )}
                            )
                        )
                        
                        
                    ### Save budget vs accuracy for AL ### 
                    data = np.rint(100 * np.array(AL_result['budget_usage_hist'])).astype(int)
                    sensors = np.rint(100 * np.array(AL_result['sensor_usage_hist'])).astype(int)
                    streamtimes = np.rint(100 * np.array(AL_result['streamtime_usage_hist'])).astype(int)
                    val_loss_hist = AL_result['val_loss_hist']
                    accuracy = np.rint(100 * (1 - np.minimum(1, val_loss_hist / RF_loss))).astype(int)
                    
                    col_name_data = '{} {} {} data'.format(
                        pred_type, 
                        AL_variable, 
                        method
                    )
                    col_name_sensors = '{} {} {} sensors'.format(
                        pred_type, 
                        AL_variable, 
                        method
                    )
                    col_name_streamtimes = '{} {} {} streamtimes'.format(
                        pred_type, 
                        AL_variable, 
                        method
                    )
                    col_name_accuracy = '{} {} {} accuracy'.format(
                        pred_type, 
                        AL_variable, 
                        method 
                    )
                    
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_data: pd.Series(data)})
                    )
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_sensors: pd.Series(sensors)})
                    )
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_streamtimes: pd.Series(streamtimes)})
                    )
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_accuracy: pd.Series(accuracy)})
                    )
                    
                    picked_times_index_hist = AL_result['picked_times_index_hist']
                    picked_spaces_index_hist = AL_result['picked_spaces_index_hist']
                    picked_inf_score_hist = AL_result['picked_inf_score_hist']
                    
                    
                    for iteration in range(n_iterations):
                        picked_times_list = picked_times_index_hist[iteration]
                        picked_spaces_list = picked_spaces_index_hist[iteration]
                        
                        col_name_times = '{} {} {} - iter {} time'.format(
                            pred_type, 
                            AL_variable, 
                            method,
                            iteration
                        )
                        col_name_spaces = '{} {} {} - iter {} space'.format(
                            pred_type, 
                            AL_variable, 
                            method,
                            iteration
                        )
                        
                        spacetime_df_list.append(
                            pd.DataFrame({col_name_times: pd.Series(picked_times_list)})
                        )
                        spacetime_df_list.append(
                            pd.DataFrame({col_name_spaces: pd.Series(picked_spaces_list)})
                        )
                        
                        # check if list is empty, in this case it is 'rnd d_c' method
                        # where no information score is available
                        if method != 'PL' and method != 'rnd d_c':
                        
                            picked_scores_list = picked_inf_score_hist[iteration] 
                            col_name_scores = '{} {} {} - iter {} score'.format(
                                pred_type, 
                                AL_variable, 
                                method,
                                iteration
                            )
                            spacetime_df_list.append(
                                pd.DataFrame({col_name_scores: pd.Series(picked_scores_list)})
                            )

            # create the index column
            df_index = [
                'n_iterations',
                't_iterations',
                'budget_usage',
                'sensor_usage',
                'streamtime_usage',
                'RF_loss',
                'test_loss',
            ]
            
            # concatenate the list of all DataFrames to final (empty) Frame
            result_df = pd.concat(df_list, axis=1)
            
            for i in range(len(result_df) - len(df_index)):
                df_index.append(i)

            # set the index column
            result_df.index = df_index

            # save results to a CSV file
            result_df.to_csv(path_to_results_file)
            
            if HYPER.TEST_SEQUENCE_IMPORTANCE:
                df_index = [
                    'test_loss',
                ]
                
                seqimportance_df = pd.concat(df_list_seqimportance, axis=1)
                
                for i in range(len(seqimportance_df) - len(df_index)):
                    df_index.append(i)
                    
                seqimportance_df.index = df_index
                seqimportance_df.to_csv(path_to_seqimportance_file)
            
            budgetvsaccuracy_df = pd.concat(budgetvsaccuracy_df_list, axis=1)
            budgetvsaccuracy_df.to_csv(path_to_budgetvsaccuracy_file)
            
            spacetime_df = pd.concat(spacetime_df_list, axis=1)
            spacetime_df.to_csv(path_to_spacetime_file)
            

def save_hyper_params(HYPER, raw_data):

    """ Saves all hyper parameter values which are used for calculating these 
    results.
    """

    if HYPER.SAVE_HYPER_PARAMS:

        for pred_type in HYPER.PRED_LIST_ACT_LRN:
            saving_path = raw_data.path_to_AL_results + pred_type + '/'
            
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)
                
            saving_path += 'hyper.csv'

            # create empty DataFrame
            hyper_df = pd.DataFrame()
            df_list = []
            
            # general parameters
            df_list.append(
                pd.DataFrame({'private_data_access': pd.Series(HYPER.PRIVATE_DATA_ACCESS)})
            )
            df_list.append(
                pd.DataFrame({'test_sequence_importance': pd.Series(
                    HYPER.TEST_SEQUENCE_IMPORTANCE
                )})
            )
            df_list.append(
                pd.DataFrame({'save_act_lrn_results': pd.Series(
                    HYPER.SAVE_ACT_LRN_RESULTS
                )})
            )
            df_list.append(
                pd.DataFrame({'save_hyper_params': pd.Series(
                    HYPER.SAVE_HYPER_PARAMS
                )})
            )
            df_list.append(
                pd.DataFrame({'save_act_lrn_models': pd.Series(
                    HYPER.SAVE_ACT_LRN_MODELS
                )})
            )
            df_list.append(
                pd.DataFrame({'save_act_lrn_test_sample': pd.Series(
                    HYPER.SAVE_ACT_LRN_TEST_SAMPLE
                )})
            )
            
            # active learning algorithm parameters
            df_list.append(
                pd.DataFrame({'pred_list_act_lrn': pd.Series(
                    HYPER.PRED_LIST_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'query_variants_act_lrn': pd.Series(
                    HYPER.QUERY_VARIANTS_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'query_variables_act_lrn': pd.Series(
                    HYPER.QUERY_VARIABLES_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'extend_train_data_act_lrn': pd.Series(
                    HYPER.EXTEND_TRAIN_DATA_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'upd_val_data_act_lrn': pd.Series(
                    HYPER.UPD_VAL_DATA_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'red_cand_data_act_lrn': pd.Series(
                    HYPER.RED_CAND_DATA_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'n_iter_act_lrn': pd.Series(
                    HYPER.N_ITER_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'data_budget_act_lrn': pd.Series(
                    HYPER.DATA_BUDGET_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'points_per_cluster_act_lrn': pd.Series(
                    HYPER.POINTS_PER_CLUSTER_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'epochs_act_lrn': pd.Series(
                    HYPER.EPOCHS_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'patience_act_lrn': pd.Series(
                    HYPER.PATIENCE_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'distance_metric_act_lrn': pd.Series(
                    HYPER.DISTANCE_METRIC_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'cluster_ethod_act_lrn': pd.Series(
                    HYPER.CLUSTER_METHOD_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'cand_subsample_act_lrn': pd.Series(
                    HYPER.CAND_SUBSAMPLE_ACT_LRN
                )})
            )

            # problem setup parameters
            df_list.append(
                pd.DataFrame({'problem_type': pd.Series(
                    HYPER.PROBLEM_TYPE
                )})
            )
            df_list.append(
                pd.DataFrame({'regression_classes': pd.Series(
                    HYPER.REGRESSION_CLASSES
                )})
            )
            df_list.append(
                pd.DataFrame({'labels': pd.Series(
                    HYPER.LABELS
                )})
            )
            df_list.append(
                pd.DataFrame({'profile_years': pd.Series(
                    HYPER.PROFILE_YEARS
                )})
            )
            df_list.append(
                pd.DataFrame({'profile_set': pd.Series(
                    HYPER.PROFILE_SET
                )})
            )
            df_list.append(
                pd.DataFrame({'profiles_per_year': pd.Series(
                    HYPER.PROFILES_PER_YEAR
                )})
            )
            df_list.append(
                pd.DataFrame({'points_per_profile': pd.Series(
                    HYPER.POINTS_PER_PROFILE
                )})
            )
            df_list.append(
                pd.DataFrame({'prediction_window': pd.Series(
                    HYPER.PREDICTION_WINDOW
                )})
            )
            df_list.append(
                pd.DataFrame({'train_split': pd.Series(
                    HYPER.TRAIN_SPLIT
                )})
            )
            df_list.append(
                pd.DataFrame({'test_split': pd.Series(
                    HYPER.TEST_SPLIT
                )})
            )
            df_list.append(
                pd.DataFrame({'split_intervals': pd.Series(
                    HYPER.SPLIT_INTERAVALS
                )})
            )

            # training and prediction model parameters
            df_list.append(
                pd.DataFrame({'epochs': pd.Series(
                    HYPER.EPOCHS
                )})
            )
            df_list.append(
                pd.DataFrame({'patience': pd.Series(
                    HYPER.PATIENCE
                )})
            )
            df_list.append(
                pd.DataFrame({'batch_size': pd.Series(
                    HYPER.BATCH_SIZE
                )})
            )
            df_list.append(
                pd.DataFrame({'encoder_layers': pd.Series(
                    HYPER.ENCODER_LAYERS
                )})
            )
            df_list.append(
                pd.DataFrame({'encoding_nodes_x_t': pd.Series(
                    HYPER.ENCODING_NODES_X_t
                )})
            )
            df_list.append(
                pd.DataFrame({'encoding_nodes_x_s': pd.Series(
                    HYPER.ENCODING_NODES_X_s
                )})
            )
            df_list.append(
                pd.DataFrame({'encoding_nodes_x_st': pd.Series(
                    HYPER.ENCODING_NODES_X_st
                )})
            )
            df_list.append(
                pd.DataFrame({'encoding_nodes_x_joint': pd.Series(
                    HYPER.ENCODING_NODES_X_joint
                )})
            )
            df_list.append(
                pd.DataFrame({'encoding_activation': pd.Series(
                    HYPER.ENCODING_ACTIVATION
                )})
            )
            df_list.append(
                pd.DataFrame({'network_layers': pd.Series(
                    HYPER.NETWORK_LAYERS
                )})
            )
            df_list.append(
                pd.DataFrame({'nodes_per_layer_dense': pd.Series(
                    HYPER.NODES_PER_LAYER_DENSE
                )})
            )
            df_list.append(
                pd.DataFrame({'filters_per_layer_cnn': pd.Series(
                    HYPER.FILTERS_PER_LAYER_CNN
                )})
            )
            df_list.append(
                pd.DataFrame({'states_per_layer_lstm': pd.Series(
                    HYPER.STATES_PER_LAYER_LSTM
                )})
            )
            df_list.append(
                pd.DataFrame({'layer_type_x_st': pd.Series(
                    HYPER.LAYER_TYPE_X_ST
                )})
            )
            df_list.append(
                pd.DataFrame({'dense_activation': pd.Series(
                    HYPER.DENSE_ACTIVATION
                )})
            )
            df_list.append(
                pd.DataFrame({'cnn_activation': pd.Series(
                    HYPER.CNN_ACTIVATION
                )})
            )
            df_list.append(
                pd.DataFrame({'lstm_activation': pd.Series(
                    HYPER.LSTM_ACTIVATION
                )})
            )
            df_list.append(
                pd.DataFrame({'initialization_method': pd.Series(
                    HYPER.INITIALIZATION_METHOD
                )})
            )
            df_list.append(
                pd.DataFrame({'initialization_method_lstm': pd.Series(
                    HYPER.INITIALIZATION_METHOD_LSTM
                )})
            )
            df_list.append(
                pd.DataFrame({'batch_normalization': pd.Series(
                    HYPER.BATCH_NORMALIZATION
                )})
            )
            df_list.append(
                pd.DataFrame({'regularizer': pd.Series(
                    HYPER.REGULARIZER
                )})
            )

            # feature parameters
            df_list.append(
                pd.DataFrame({'timestamp_data': pd.Series(
                    HYPER.TIMESTAMP_DATA
                )})
            )
            df_list.append(
                pd.DataFrame({'time_encoding': pd.Series(
                    HYPER.TIME_ENCODING
                )})
            )
            df_list.append(
                pd.DataFrame({'spatial_features': pd.Series(
                    HYPER.SPATIAL_FEATURES
                )})
            )
            df_list.append(
                pd.DataFrame({'histo_bins': pd.Series(
                    HYPER.HISTO_BINS
                )})
            )
            df_list.append(
                pd.DataFrame({'grey_scale': pd.Series(
                    HYPER.GREY_SCALE
                )})
            )
            df_list.append(
                pd.DataFrame({'down_scale_building_images': pd.Series(
                    HYPER.DOWN_SCALE_BUILDING_IMAGES
                )})
            )
            df_list.append(
                pd.DataFrame({'meteo_types': pd.Series(
                    HYPER.METEO_TYPES
                )})
            )
            df_list.append(
                pd.DataFrame({'history_window_meteo': pd.Series(
                    HYPER.HISTORY_WINDOW_METEO
                )})
            )
            df_list.append(
                pd.DataFrame({'normalization': pd.Series(
                    HYPER.NORMALIZATION
                )})
            )
            df_list.append(
                pd.DataFrame({'standardization': pd.Series(
                    HYPER.STANDARDIZATION
                )})
            )

            # concatenate the list of all DataFrames to final (empty) Frame
            hyper_df = pd.concat(df_list, axis=1)

            # save results to a CSV file
            hyper_df.to_csv(saving_path)


def save_act_lrn_test_sample(
    HYPER, 
    raw_data, 
    AL_result_list, 
    PL_result_list):

    """ Saves a random sample of 1,000 data points from the candidate data pool 
    which were not seen by the actively trained prediction models.
    """

    if HYPER.SAVE_ACT_LRN_TEST_SAMPLE:
        if HYPER.SPATIAL_FEATURES == 'image':
            print('Feature not available')
            return

        for pred_type in HYPER.PRED_LIST_ACT_LRN:
            saving_path = raw_data.path_to_AL_test_samples + pred_type + '/'
            
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)

            # get method_result_list of currently iterated prediction type
            var_result_list = AL_result_list[pred_type]

            # get random results
            PL_results = PL_result_list[pred_type]
            test_data = PL_results['test_data']

            X_t = test_data.X_t
            X_s = test_data.X_s
            X_st = test_data.X_st
            Y = test_data.Y
            
            if HYPER.SPATIAL_FEATURES != 'image':
                X_s1 = test_data.X_s1
            else:
                X_s1 = 0

            saving_list = ['X_t', 'X_s', 'X_s1', 'X_st', 'Y']
            for var in saving_list:
                path_to_var = (
                    saving_path + 'PL_' + var
                )
                command = 'np.save(path_to_var, ' + var + ')'
                exec(command)

            for AL_variable in HYPER.QUERY_VARIABLES_ACT_LRN:
                # get variable result list
                method_result_list = var_result_list[AL_variable]

                for method in HYPER.QUERY_VARIANTS_ACT_LRN:

                    AL_result = method_result_list[method]
                    test_data = AL_result['test_data']

                    X_t = test_data.X_t
                    X_s = test_data.X_s
                    X_st = test_data.X_st
                    Y = test_data.Y
                    
                    if HYPER.SPATIAL_FEATURES != 'image':
                        X_s1 = test_data.X_s1
                    else:
                        X_s1 = 0

                    saving_list = ['X_t', 'X_s', 'X_s1', 'X_st', 'Y']
                    for var in saving_list:
                        path_to_var = (
                            saving_path
                            + AL_variable
                            + ' '
                            + method
                            + ' '
                            + var
                        )
                        command = 'np.save(path_to_var, ' + var + ')'
                        exec(command)
