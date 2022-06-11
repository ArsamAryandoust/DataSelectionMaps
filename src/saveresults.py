import os
import shutil
import numpy as np
import pandas as pd


def saveallresults(
    HYPER,
    raw_data,
    RF_result, 
    AL_result_dict, 
    PL_result_dict

):

    """ Calls all other functions to save generated results """
    
    # remove temporary saved encoder weights
    shutil.rmtree(raw_data.path_to_tmp_encoder_weights)
    
    if HYPER.SAVE_RESULTS:
        # save active learning results
        save_act_lrn_results(
            HYPER, 
            raw_data, 
            RF_result, 
            AL_result_dict, 
            PL_result_dict
        )

        if HYPER.TEST_EXPERIMENT_CHOICE == 'main_experiments':
            # save hyper parameters
            save_hyper_params(
                HYPER, 
                raw_data
            )

            # save the prediction models
            save_act_lrn_models(
                HYPER, 
                raw_data, 
                AL_result_dict, 
                PL_result_dict
            )

            # save the test data sample
            save_act_lrn_test_sample(
                HYPER, 
                raw_data, 
                AL_result_dict, 
                PL_result_dict
            )


def save_act_lrn_models(
    HYPER, 
    raw_data, 
    AL_result_dict, 
    PL_result_dict
):

    """ Saves the actively trained prediction models. """

    for pred_type in HYPER.PRED_LIST_ACT_LRN:

        # get method_result_list of currently iterated prediction type
        var_result_dict = AL_result_dict[pred_type]

        # get random results
        PL_result = PL_result_dict[pred_type]

        prediction_model = PL_result['prediction_model']

        # create the full path for saving random  prediction model
        saving_path = raw_data.path_to_AL_models + pred_type + '/'
    
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
            
        path_to_model = saving_path + 'PL.h5'

        # save currently iterated model
        prediction_model.save(path_to_model)

        for AL_variable in HYPER.QUERY_VARIABLES_ACT_LRN:

            # get variable result list
            method_result_dict = var_result_dict[AL_variable]

            for method in HYPER.QUERY_VARIANTS_ACT_LRN:

                # get result object and prediction model
                AL_result = method_result_dict[method]
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
    RF_result, 
    AL_result_dict, 
    PL_result_dict
):

    """ Saves the active learning results, including number of iterations used, 
    time used for each iteration, share of data budget used, share of sensor 
    budget used, share of stream time budget used, testing loss baseline loss 
    and passive learning benchmark histories, validation histories and training 
    histories.
    """

    for pred_type in HYPER.PRED_LIST_ACT_LRN:
        saving_path = raw_data.path_to_AL_results + pred_type + '/'
        
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
            
        
        df_list_main = []
        path_to_results_file = saving_path + 'results.csv'
        
        RF_loss = RF_result[pred_type]
        var_result_dict = AL_result_dict[pred_type]

        ### Save PL results ###
        PL_result = PL_result_dict[pred_type]

        n_iterations = HYPER.N_ITER_ACT_LRN
        t_iter_avg = (
            sum(PL_result['iter_time_hist']) 
            / len(PL_result['iter_time_hist'])
        )
        budget_usage = PL_result['budget_usage_hist'][-1]
        sensor_usage = PL_result['sensor_usage_hist'][-1]
        streamtime_usage = PL_result['streamtime_usage_hist'][-1]
        test_loss = PL_result['test_loss']

        train_hist = PL_result['train_hist']
        val_hist = PL_result['val_hist']

        col_name_train = '{} {} {} train'.format(pred_type, None, 'PL')
        col_name_val = '{} {} {} val'.format(pred_type, None, 'PL')

        df_index_base = [
            't_iter_avg',
            'budget_usage',
            'sensor_usage',
            'streamtime_usage',
            'test_loss',
            'RF_loss',
            'cand_subsample_rate',
            'points_percluster_rate'
        ]
        
        meta_entry = np.array(
            [
                t_iter_avg,
                budget_usage,
                sensor_usage,
                streamtime_usage,
                test_loss,
                RF_loss,
                '-',
                '-'
            ]
        )
        entry_train = np.concatenate((meta_entry, train_hist))
        entry_val = np.concatenate((meta_entry, val_hist))

        df_list_main.append(
            pd.DataFrame({col_name_train: pd.Series(entry_train)})
        )
        df_list_main.append(
            pd.DataFrame({col_name_val: pd.Series(entry_val)})
        )
        
        ### Prepare query by coordinate imporatance for AL ### 
        df_list_querybycoordinate = df_list_main.copy()
        path_to_querybycoordinate_file = (
            saving_path 
            + 'heuristic_querybycoordinate.csv'
        )
        
        ### Prepare sequence imporatance for AL ### 
        df_list_seqimportance = []
        path_to_seqimportance_file = (
            saving_path 
            + 'sequence_importance.csv'
        )
            
        ### Prepare points per cluster imporatance for AL ### 
        df_list_pointspercluster = []
        path_to_pointspercluster_file = (
            saving_path 
            + 'heuristic_pointspercluster.csv'
        )
        
        ### Prepare subsample imporatance for AL ###
        df_list_subsample = []
        path_to_subsample_file = (
            saving_path 
            + 'heuristic_subsampling.csv'
        )

        ### Prepare budget vs accuracy for PL ###
        df_list_budgetvsaccuracy = []
        path_to_budgetvsaccuracy_file = (
            saving_path 
            + 'budget_vs_accuracy.csv'
        )
        
        ### Prepare picked times and spaces for PL ###
        df_list_spacetime = []
        path_to_spacetime_file = (
            saving_path 
            + 'spacetime_selection.csv'
        )
        
        
        ### Create entries for budgets vs accuracy results ###
        data = np.rint(
            100 * np.array(PL_result['budget_usage_hist'])
        ).astype(int)
        sensors = np.rint(
            100 * np.array(PL_result['sensor_usage_hist'])
        ).astype(int)
        streamtimes = np.rint(
            100 * np.array(PL_result['streamtime_usage_hist'])
        ).astype(int)
        accuracy = np.rint(
            100 * (1 - np.minimum(1, PL_result['val_loss_hist'] / RF_loss))
        ).astype(int)
        
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

        df_list_budgetvsaccuracy.append(
            pd.DataFrame({col_name_data: pd.Series(data)})
        )
        df_list_budgetvsaccuracy.append(
            pd.DataFrame({col_name_sensors: pd.Series(sensors)})
        )
        df_list_budgetvsaccuracy.append(
            pd.DataFrame({col_name_streamtimes: pd.Series(streamtimes)})
        )
        df_list_budgetvsaccuracy.append(
            pd.DataFrame({col_name_accuracy: pd.Series(accuracy)})
        )
        
        ### Create entries for spacetime results ###
        picked_times_index_hist = PL_result['picked_times_index_hist']
        picked_spaces_index_hist = PL_result['picked_spaces_index_hist']
        initial_sensors_list = PL_result['initial_sensors_list']
        
        col_name_initial_sensors = '{} - initial sensors'.format(
            pred_type
        )
        
        df_list_spacetime.append(
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
            
            df_list_spacetime.append(
                pd.DataFrame({col_name_times: pd.Series(picked_times_list)})
            )
            df_list_spacetime.append(
                pd.DataFrame({col_name_spaces: pd.Series(picked_spaces_list)})
            )
        
        
        for AL_variable in HYPER.QUERY_VARIABLES_ACT_LRN:
            method_result_dict = var_result_dict[AL_variable]

            for method in HYPER.QUERY_VARIANTS_ACT_LRN:
                AL_result = method_result_dict[method]
                
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
                
                ### Save heuristics test results for query by coordinate ###
                if HYPER.TEST_EXPERIMENT_CHOICE == 'querybycoordinate_importance':
                    heuristics_result_list = AL_result['heuristics_querybycoordinate']
                    
                    for heuristic_dict in heuristics_result_list:
                        cand_subsample_rate = (
                            heuristic_dict['cand_subsample_rate']
                        )
                        points_percluster_rate = (
                            heuristic_dict['points_percluster_rate']
                        )
                        t_iter_avg = (
                            sum(heuristic_dict['iter_time_hist']) 
                            / len(heuristic_dict['iter_time_hist'])
                        )
                        budget_usage = (
                            heuristic_dict['budget_usage_hist'][-1]
                        )
                        sensor_usage = (
                            heuristic_dict['sensor_usage_hist'][-1]
                        )
                        streamtime_usage = (
                            heuristic_dict['streamtime_usage_hist'][-1]
                        )
                        test_loss = (
                            heuristic_dict['test_loss']
                        )
                        train_hist = (
                            heuristic_dict['train_hist']
                        )
                        val_hist = (
                            heuristic_dict['val_hist']
                        )
                        
                        meta_entry = np.array(
                            [
                                t_iter_avg,
                                budget_usage,
                                sensor_usage,
                                streamtime_usage,
                                test_loss,
                                RF_loss,
                                cand_subsample_rate,
                                points_percluster_rate
                            ]
                        )
                        
                        entry_train = np.concatenate(
                            (
                                meta_entry, 
                                train_hist
                            )
                        )
                        entry_val = np.concatenate(
                            (
                                meta_entry, 
                                val_hist
                            )
                        )
                    
                        df_list_querybycoordinate.append(
                            pd.DataFrame(
                                {col_name_train: pd.Series(
                                    entry_train
                                )}
                            )
                        )
                        df_list_querybycoordinate.append(
                            pd.DataFrame(
                                {col_name_val: pd.Series(
                                    entry_val
                                )}
                            )
                        )
                    
                ### Save main AL results ### 
                elif HYPER.TEST_EXPERIMENT_CHOICE == 'main_experiments':
                
                    t_iter_avg = sum(AL_result['iter_time_hist']) / len(AL_result['iter_time_hist'])
                    budget_usage = AL_result['budget_usage_hist'][-1]
                    sensor_usage = AL_result['sensor_usage_hist'][-1]
                    streamtime_usage = AL_result['streamtime_usage_hist'][-1]
                    test_loss = AL_result['test_loss']
                    cand_subsample_rate = AL_result['cand_subsample_rate']
                    points_percluster_rate = AL_result['points_percluster_rate']
                    train_hist = AL_result['train_hist']
                    val_hist = AL_result['val_hist']
                    
                    meta_entry = np.array(
                        [
                            t_iter_avg,
                            budget_usage,
                            sensor_usage,
                            streamtime_usage,
                            test_loss,
                            RF_loss,
                            cand_subsample_rate,
                            points_percluster_rate
                        ]
                    )
                    entry_train = np.concatenate((meta_entry, train_hist))
                    entry_val = np.concatenate((meta_entry, val_hist))

                    df_list_main.append(
                        pd.DataFrame({col_name_train: pd.Series(entry_train)})
                    )
                    df_list_main.append(
                        pd.DataFrame({col_name_val: pd.Series(entry_val)})
                    )
                    
                    
                    ### Save budget vs accuracy for ADL ###
                    data = np.rint(
                        100 * np.array(AL_result['budget_usage_hist'])
                    ).astype(int)
                    sensors = np.rint(
                        100 * np.array(AL_result['sensor_usage_hist'])
                    ).astype(int)
                    streamtimes = np.rint(
                        100 * np.array(AL_result['streamtime_usage_hist'])
                    ).astype(int)
                    accuracy = np.rint(
                        100 * (1 - np.minimum(1, AL_result['val_loss_hist'] / RF_loss))
                    ).astype(int)
                    
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
                    
                    df_list_budgetvsaccuracy.append(
                        pd.DataFrame({col_name_data: pd.Series(data)})
                    )
                    df_list_budgetvsaccuracy.append(
                        pd.DataFrame({col_name_sensors: pd.Series(sensors)})
                    )
                    df_list_budgetvsaccuracy.append(
                        pd.DataFrame({col_name_streamtimes: pd.Series(streamtimes)})
                    )
                    df_list_budgetvsaccuracy.append(
                        pd.DataFrame({col_name_accuracy: pd.Series(accuracy)})
                    )
                    
                    ### Save space time selection for ADL ###
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
                        
                        df_list_spacetime.append(
                            pd.DataFrame({col_name_times: pd.Series(picked_times_list)})
                        )
                        df_list_spacetime.append(
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
                            df_list_spacetime.append(
                                pd.DataFrame({col_name_scores: pd.Series(picked_scores_list)})
                            )
                
                ### Save sequence importance for AL ### 
                elif HYPER.TEST_EXPERIMENT_CHOICE == 'sequence_importance':
                    
                    AL_sequence = AL_result['AL_sequence']
                    AL_train_loss_seqimportance = (
                        AL_sequence['train_hist']
                    )
                    AL_val_loss_seqimportance = (
                        AL_sequence['val_hist']
                    )
                    AL_test_loss_seqimportance = (
                        AL_sequence['test_loss']
                    )
                    AL_meta_entry = np.array(
                        [
                            AL_test_loss_seqimportance
                        ]
                    )
                    AL_entry_train_seqimportance = np.concatenate(
                        (
                            AL_meta_entry, 
                            AL_train_loss_seqimportance
                        )
                    )
                    AL_entry_val_seqimportance = np.concatenate(
                        (
                            AL_meta_entry, 
                            AL_val_loss_seqimportance
                        )
                    )
                    
                    random_sequence = AL_result['random_sequence']
                    random_train_loss_seqimportance = (
                        random_sequence['train_hist']
                    )
                    random_val_loss_seqimportance = (
                        random_sequence['val_hist']
                    )
                    random_test_loss_seqimportance = (
                        random_sequence['test_loss']
                    )
                    random_meta_entry = np.array(
                        [
                            random_test_loss_seqimportance
                        ]
                    )
                    random_entry_train_seqimportance = np.concatenate(
                        (
                            random_meta_entry, 
                            random_train_loss_seqimportance
                        )
                    )
                    random_entry_val_seqimportance = np.concatenate(
                        (
                            random_meta_entry, 
                            random_val_loss_seqimportance
                        )
                    )
                    random_col_name_train = 'random ' + col_name_train
                    random_col_name_val = 'random ' + col_name_val
                    
                    df_list_seqimportance.append(
                        pd.DataFrame(
                            {col_name_train: pd.Series(
                                AL_entry_train_seqimportance
                            )}
                        )
                    )
                    df_list_seqimportance.append(
                        pd.DataFrame(
                            {col_name_val: pd.Series(
                                AL_entry_val_seqimportance
                            )}
                        )
                    )
                    df_list_seqimportance.append(
                        pd.DataFrame(
                            {random_col_name_train: pd.Series(
                                random_entry_train_seqimportance
                            )}
                        )
                    )
                    df_list_seqimportance.append(
                        pd.DataFrame(
                            {random_col_name_val: pd.Series(
                                random_entry_val_seqimportance
                            )}
                        )
                    )
                    
                    
                ### Save subsample importance for AL ### 
                elif HYPER.TEST_EXPERIMENT_CHOICE == 'subsample_importance':
                    heuristics_list = AL_result['heuristics_subsample']
                    
                    for heuristic_dict in heuristics_list:
                        cand_subsample_rate = (
                            heuristic_dict['cand_subsample_rate']
                        )
                        points_percluster_rate = (
                            heuristic_dict['points_percluster_rate']
                        )
                        t_iter_avg = (
                            sum(heuristic_dict['iter_time_hist']) 
                            / len(heuristic_dict['iter_time_hist'])
                        )
                        budget_usage = (
                            heuristic_dict['budget_usage_hist'][-1]
                        )
                        sensor_usage = (
                            heuristic_dict['sensor_usage_hist'][-1]
                        )
                        streamtime_usage = (
                            heuristic_dict['streamtime_usage_hist'][-1]
                        )
                        test_loss = (
                            heuristic_dict['test_loss']
                        )
                        train_hist = (
                            heuristic_dict['train_hist']
                        )
                        val_hist = (
                            heuristic_dict['val_hist']
                        )
                        
                        meta_entry = np.array(
                            [
                                t_iter_avg,
                                budget_usage,
                                sensor_usage,
                                streamtime_usage,
                                test_loss,
                                RF_loss,
                                cand_subsample_rate,
                                points_percluster_rate
                            ]
                        )
                        
                        entry_train = np.concatenate(
                            (
                                meta_entry, 
                                train_hist
                            )
                        )
                        entry_val = np.concatenate(
                            (
                                meta_entry, 
                                val_hist
                            )
                        )
                        
                ### Save points per cluster importance for AL ### 
                elif HYPER.TEST_EXPERIMENT_CHOICE == 'pointspercluster_importance':
                    heuristics_list = AL_result['heuristics_pointspercluster']
                    
                    for heuristic_dict in heuristics_list:
                        cand_subsample_rate = (
                            heuristic_dict['cand_subsample_rate']
                        )
                        points_percluster_rate = (
                            heuristic_dict['points_percluster_rate']
                        )
                        t_iter_avg = (
                            sum(heuristic_dict['iter_time_hist']) 
                            / len(heuristic_dict['iter_time_hist'])
                        )
                        budget_usage = (
                            heuristic_dict['budget_usage_hist'][-1]
                        )
                        sensor_usage = (
                            heuristic_dict['sensor_usage_hist'][-1]
                        )
                        streamtime_usage = (
                            heuristic_dict['streamtime_usage_hist'][-1]
                        )
                        test_loss = (
                            heuristic_dict['test_loss']
                        )
                        train_hist = (
                            heuristic_dict['train_hist']
                        )
                        val_hist = (
                            heuristic_dict['val_hist']
                        )
                        
                        meta_entry = np.array(
                            [
                                t_iter_avg,
                                budget_usage,
                                sensor_usage,
                                streamtime_usage,
                                test_loss,
                                RF_loss,
                                cand_subsample_rate,
                                points_percluster_rate
                            ]
                        )
                        
                        entry_train = np.concatenate(
                            (
                                meta_entry, 
                                train_hist
                            )
                        )
                        entry_val = np.concatenate(
                            (
                                meta_entry, 
                                val_hist
                            )
                        )
                        
                        df_list_pointspercluster.append(
                            pd.DataFrame(
                                {col_name_train: pd.Series(
                                    entry_train
                                )}
                            )
                        )
                        df_list_pointspercluster.append(
                            pd.DataFrame(
                                {col_name_val: pd.Series(
                                    entry_val
                                )}
                            )
                        )
                    

        ### Save main results ###
        if HYPER.TEST_EXPERIMENT_CHOICE == 'main_experiments':
            df_index = df_index_base.copy()
            result_df = pd.concat(df_list_main, axis=1)
            for i in range(len(result_df) - len(df_index)):
                df_index.append(i)
            result_df.index = df_index
            result_df.to_csv(path_to_results_file)
            
            # Save results fur budget vs. accuracy
            result_df = pd.concat(df_list_budgetvsaccuracy, axis=1)
            result_df.to_csv(path_to_budgetvsaccuracy_file)
            
            # Save results for spacetime data points selection
            result_df = pd.concat(df_list_spacetime, axis=1)
            result_df.to_csv(path_to_spacetime_file)
        
        ### Save query by coordinate results ###
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'querybycoordinate_importance':
            df_index = df_index_base.copy()
            result_df = pd.concat(df_list_querybycoordinate, axis=1)
            for i in range(len(result_df) - len(df_index)):
                df_index.append(i)
            result_df.index = df_index
            result_df.to_csv(path_to_querybycoordinate_file)
        
        ### Save sequence importance results ###
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'sequence_importance':
            df_index = ['test_loss']
            result_df = pd.concat(df_list_seqimportance, axis=1)
            for i in range(len(result_df) - len(df_index)):
                df_index.append(i)
            result_df.index = df_index
            result_df.to_csv(path_to_seqimportance_file)
        
        ### Save results for subsample test ###
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'subsample_importance':
            df_index = df_index_base.copy() 
            df_index.pop()
            result_df = pd.concat(df_list_subsample, axis=1)
            for i in range(len(result_df) - len(df_index)):
                df_index.append(i)
            result_df.index = df_index
            result_df.to_csv(path_to_subsample_file)
        
        ### Save results for pointspercluster test ###
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'pointspercluster_importance':   
            df_index = df_index_base.copy() 
            df_index.pop()
            result_df = pd.concat(df_list_pointspercluster, axis=1)
            for i in range(len(result_df) - len(df_index)):
                df_index.append(i)
            result_df.index = df_index
            result_df.to_csv(path_to_pointspercluster_file)
        

def save_hyper_params(HYPER, raw_data):

    """ Saves all hyper parameter values which are used for calculating these 
    results.
    """

    
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
            pd.DataFrame({'private_data_access': pd.Series(
                HYPER.PRIVATE_DATA_ACCESS
            )})
        )
        df_list.append(
            pd.DataFrame({'test_experiment_choice': pd.Series(
                HYPER.TEST_EXPERIMENT_CHOICE
            )})
        )
        df_list.append(
            pd.DataFrame({'save_results': pd.Series(
                HYPER.SAVE_RESULTS
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
            pd.DataFrame({'cand_subsample_act_lrn': pd.Series(
                HYPER.CAND_SUBSAMPLE_ACT_LRN
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
            pd.DataFrame({'cluster_method_act_lrn': pd.Series(
                HYPER.CLUSTER_METHOD_ACT_LRN
            )})
        )

        # problem setup parameters
        df_list.append(
            pd.DataFrame({'problem_type': pd.Series(
                HYPER.PROBLEM_TYPE
            )})
        )
        df_list.append(
            pd.DataFrame({'regression_loss_name': pd.Series(
                HYPER.REGRESSION_LOSS_NAME
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
    AL_result_dict, 
    PL_result_dict):

    """ Saves a random sample of 1,000 data points from the candidate data pool 
    which were not seen by the actively trained prediction models.
    """

    if HYPER.SPATIAL_FEATURES == 'image':
        print('Feature not available')
        return

    for pred_type in HYPER.PRED_LIST_ACT_LRN:
        saving_path = raw_data.path_to_AL_test_samples + pred_type + '/'
        
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)

        # get method_result_list of currently iterated prediction type
        var_result_dict = AL_result_dict[pred_type]

        # get random results
        PL_result = PL_result_dict[pred_type]
        test_data = PL_result['test_data']

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
            method_result_dict = var_result_dict[AL_variable]

            for method in HYPER.QUERY_VARIANTS_ACT_LRN:

                AL_result = method_result_dict[method]
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
