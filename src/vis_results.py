import os
import pandas as pd
import numpy as np

class HyperParameterVisualizing:

    """ Keeps hyper parameters together for visualizing results
    """
    
    PATH_TO_RESULTS = '../results/'
    SUB_TITLE_LIST = [
        'a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.',
        'i.', 'j.', 'k.', 'l.', 'm.', 'n.', 'o.', 'p.',
        'q.', 'r.', 's.', 't.', 'u.', 'v.', 'w.', 'x.'
    ]
    WIDTH_FACTOR = 8
    FONTSIZE = 20
    
    
def test_hyper(HYPER_VIS):

    "Checks if all experiments were calculated on matching hyper parameters."
    
    profile_type_list = os.listdir(HYPER_VIS.PATH_TO_RESULTS)
    counter = 0
    for profile_type in profile_type_list:
        path_to_results = HYPER_VIS.PATH_TO_RESULTS + profile_type + '/'
        pred_type_list = os.listdir(path_to_results)
        
        for pred_type in pred_type_list:
            path_to_pred = path_to_results + pred_type + '/'
            exp_type_list = os.listdir(path_to_pred)
            
            for exp_type in exp_type_list:
                path_to_exp = path_to_pred + exp_type + '/'
                result_type_list = os.listdir(path_to_exp)
                
                if 'values' in result_type_list:
                    path_to_values = path_to_exp + 'values/'
                    file_type_list = os.listdir(path_to_values)
                    
                    if 'hyper.csv' in file_type_list:
                        path_to_hyper = path_to_values + 'hyper.csv'
                        hyper_df = pd.read_csv(path_to_hyper)
                        hyper_df.drop(
                            [
                                'pred_type_act_lrn',
                                'red_cand_data_act_lrn',
                                'upd_val_data_act_lrn'
                            ],
                            axis=1,
                            inplace=True
                        )
                        counter += 1
                        if (
                            counter > 1 and
                            not hyper_df.equals(prev_hyper_df) and
                            hyper_df['profile_set'].equals(prev_hyper_df['profile_set'])
                        ):
                            print(
                                'Caution. Results were not calculated on same', 
                                'hyper parameter for:\n{}\n{}\n{}'.format(
                                    profile_type,
                                    pred_type,
                                    exp_type
                                )
                            )
                        
                        prev_hyper_df = hyper_df
                        
                        
def show_numerical_results(HYPER_VIS):

    "Summarizes and shows us the numerical results of our experiments"
    
    profile_type_list = os.listdir(HYPER_VIS.PATH_TO_RESULTS)
    for profile_type in profile_type_list:
        path_to_results = HYPER_VIS.PATH_TO_RESULTS + profile_type + '/'
        pred_type_list = os.listdir(path_to_results)
        
        for pred_type in pred_type_list:
            path_to_pred = path_to_results + pred_type + '/'
            exp_type_list = os.listdir(path_to_pred)
            
            for exp_type in exp_type_list:
                path_to_exp = path_to_pred + exp_type + '/'
                result_type_list = os.listdir(path_to_exp)
                
                if 'values' in result_type_list:
                    path_to_values = path_to_exp + 'values/'
                    file_type_list = os.listdir(path_to_values)
                    
                    if 'results.csv' in file_type_list:
                        path_to_results = path_to_values + 'results.csv'
                        results_df = pd.read_csv(path_to_results)
                        
                        results_transformed = (
                            results_df[1:6].set_index('Unnamed: 0').transpose()
                        )
                        results_transformed = (
                            results_transformed.drop(['streamtime_usage'], axis=1)
                        )
                        results_transformed['test_loss'] = (
                            results_transformed['test_loss'].values.astype(float)
                        )
                        results_transformed['RF_loss'] = (
                            results_transformed['RF_loss'].values.astype(float)
                        )
                        results_transformed['test_loss'] = (
                            results_transformed['test_loss'].values.astype(float)
                        )
                        results_transformed['sensor_usage'] = (
                            results_transformed['sensor_usage'].values.astype(float)
                        )
                        results_transformed['budget_usage'] = (
                            results_transformed['budget_usage'].values.astype(float)
                        )
                        results_transformed['accuracy'] = (
                            100 * (1 - np.minimum(1, results_transformed['test_loss'] / (
                                results_transformed['RF_loss']
                            )
                        ))).round().astype(int)
                        results_transformed['sensor_usage'] = (
                            (100 * results_transformed['sensor_usage']).round().astype(int)
                        )
                        results_transformed['budget_usage'] = (
                            (100 * results_transformed['budget_usage']).round().astype(int)
                        )
                        print(exp_type)
                        display(results_transformed)
