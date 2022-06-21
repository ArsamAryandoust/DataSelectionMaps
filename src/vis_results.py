import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

class HyperParameterVisualizing:

    """ Keeps hyper parameters together for visualizing results
    """
    
    SAVE_RESULTS = True
    PATH_TO_RESULTS = '../results/'
    PATH_TO_IMAGES = '../images/manuscript/'
    SUB_TITLE_LIST = [
        'a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.',
        'i.', 'j.', 'k.', 'l.', 'm.', 'n.', 'o.', 'p.',
        'q.', 'r.', 's.', 't.', 'u.', 'v.', 'w.', 'x.'
    ]
    WIDTH_FACTOR = 8
    FONTSIZE = 22
    LEGEND_FONTSIZE = FONTSIZE - 8
    CAND_SUBSAMPLE_TEST_LIST = [0.3, 0.5, 0.7, 1]
    POINTS_PERCLUSTER_TEST_LIST = [0, 0.25, 0.5, 1]
    POSSIBLE_QUERY_VARIABLES_ACT_LRN = ['X_t', 'X_s1', 'X_st', 'X_(t,s)', 'Y_hat_(t,s)', 'Y_(t,s)']
    POSSIBLE_QUERY_VARIANTS_ACT_LRN = ['rnd d_c', 'min d_c', 'max d_c', 'avg d_c']
    HEURISTICS_COLOR_LIST = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b',  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    
    RESULT_SUMMARY = {
        'n_rows' : 4,
        'n_cols' : 2,
        'plot_list' : [
            {
                'row': 0,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup1',
                'exp_choice' : 'main_experiments',
                'plot_type': 'train',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 0,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup1',
                'exp_choice' : 'main_experiments',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 1,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup1',
                'exp_choice' : 'main_experiments',
                'plot_type': 'train',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 1,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup1',
                'exp_choice' : 'main_experiments',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 2,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup0',
                'exp_choice' : 'main_experiments',
                'plot_type': 'train',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': [0, 3]
            },
            {
                'row': 2,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup0',
                'exp_choice' : 'main_experiments',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 3,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup0',
                'exp_choice' : 'main_experiments',
                'plot_type': 'train',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 3,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup0',
                'exp_choice' : 'main_experiments',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            }
        ],

    }
    
    HEURISTIC_SUMMARY = {
        'n_rows' : 4,
        'n_cols' : 2,
        'plot_list' : [
            {
                'row': 0,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup1',
                'exp_choice' : 'subsample_importance',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None 
            },
            {
                'row': 0,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup1',
                'exp_choice' : 'subsample_importance',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': [0, 1.4]
            },
            {
                'row': 1,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup1',
                'exp_choice' : 'pointspercluster_importance',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 1,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup1',
                'exp_choice' : 'pointspercluster_importance',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': None
            },
            {
                'row': 2,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup1',
                'exp_choice' : 'querybycoordinate_importance',
                'plot_type': 'val',
                'AL_variable': 'X_s1',
                'y_lims': [0.4, 1.9]
            },
            {
                'row': 2,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup1',
                'exp_choice' : 'querybycoordinate_importance',
                'plot_type': 'val',
                'AL_variable': 'X_s1',
                'y_lims': [0, 6]
            },
            {
                'row': 3,
                'col': 0,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta0_valup1',
                'exp_choice' : 'sequence_importance',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': [0, 1]
            },
            {
                'row': 3,
                'col': 1,
                'profile_type': 'profiles_100',
                'pred_type' : 'spatio-temporal',
                'exp_type' : 'delta1_valup1',
                'exp_choice' : 'sequence_importance',
                'plot_type': 'val',
                'AL_variable': 'Y_hat_(t,s)',
                'y_lims': [0, 1]
            }
        ],
    }
    
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
                            results_df[:6].set_index('Unnamed: 0').transpose()
                        )
                        results_transformed = (
                            results_transformed.drop(['streamtime_usage'], axis=1)
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
                        results_transformed['t_iter_avg'] = (
                            results_transformed['t_iter_avg']/results_transformed['t_iter_avg'][0]
                        ).round(1)
                        results_transformed.rename(columns={"t_iter_avg":"comp_fac"}, inplace=True)
                        
                        print(profile_type)
                        print(exp_type)
                        display(results_transformed)
                        
                        
def plot_train_val_hist(
    HYPER_VIS
):
    
    """ Plots the main experimental results
    """
    mpl.rcParams.update({'font.size': HYPER_VIS.FONTSIZE})
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
                        
                        path_to_hyper = path_to_values + 'hyper.csv'
                        hyper_df = pd.read_csv(path_to_hyper)
                        
                        path_to_figures = path_to_exp + 'figures/'
                        if not os.path.exists(path_to_figures):
                            os.mkdir(path_to_figures)
                        
                        col_name_train = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'train'
                        )
                        col_name_val = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'val'
                        )
                        
                        PL_t_iter_avg = results_df[col_name_train][0]
                        budget_usage = results_df[col_name_train][1]
                        sensor_usage = results_df[col_name_train][2]
                        PL_loss = results_df[col_name_train][4]
                        RF_loss = results_df[col_name_train][5]
                        PL_accuracy = 1 - min(1, PL_loss /RF_loss)
                        PL_train = results_df[col_name_train][8:].dropna().values
                        PL_val = results_df[col_name_val][8:].dropna().values
                        
                        legend_PL_train = 'PDL baseline:  1x comp'
                        legend_PL_val = 'PDL baseline:  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                            budget_usage, 
                            sensor_usage,
                            PL_accuracy
                        )
                        
                        query_variables_act_lrn = hyper_df['query_variables_act_lrn'].dropna()
                        query_variants_act_lrn = hyper_df['query_variants_act_lrn'].dropna()
                        
                        fig, ax = plt.subplots(
                            len(query_variables_act_lrn), 
                            2, 
                            figsize=(
                                20, 
                                len(query_variables_act_lrn) * HYPER_VIS.WIDTH_FACTOR
                            )
                        )
                        
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                                
                            plot_left.plot(
                                PL_train, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL_train
                            )
                            plot_right.plot(
                                PL_val, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL_val
                            )
                        
                        
                            if index_var == 0:
                                cols = [
                                    'Training losses \n {}'.format(AL_variable), 
                                    'Validation losses \n {}'.format(AL_variable)
                                ]
                                for axes, col in zip(ax[0], cols):
                                    axes.set_title(col)
                            else:
                                plot_left.set_title(AL_variable)
                                plot_right.set_title(AL_variable)
                        
                        
                            for index_method, AL_variant in enumerate(query_variants_act_lrn):
                
                                col_name_train = (
                                    pred_type 
                                    + ' ' 
                                    + AL_variable 
                                    + ' ' 
                                    + AL_variant 
                                    + ' train'
                                )
                                col_name_val = (
                                    pred_type 
                                    + ' ' 
                                    + AL_variable 
                                    + ' ' 
                                    + AL_variant 
                                    + ' val'
                                )

                                # get training losses for mode 1 with validation updates
                                AL_t_iter_avg = results_df[col_name_train][0]
                                budget_usage = results_df[col_name_train][1]
                                sensor_usage = results_df[col_name_train][2]
                                AL_loss = results_df[col_name_train][4]
                                RF_loss = results_df[col_name_train][5]
                                AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                                AL_train = results_df[col_name_train][8:].dropna().values
                                AL_val = results_df[col_name_val][8:].dropna().values

                                # create the legends
                                legend_train = 'ADL {}:  {}x comp'.format(
                                    AL_variant, 
                                    round(AL_t_iter_avg / PL_t_iter_avg, 1)
                                )
                                legend_val = 'ADL {}:  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                                    AL_variant, 
                                    budget_usage, 
                                    sensor_usage,
                                    AL_accuracy
                                )

                                # plot iterated training losses
                                plot_left.plot(
                                    AL_train, 
                                    label=legend_train
                                )
                                plot_right.plot(
                                    AL_val, 
                                    label=legend_val
                                )

                            # set legend
                            plot_left.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )
                            plot_right.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )

                            # set y-axis labels
                            plot_left.set_ylabel(
                                'L2 loss [kW²]', 
                            )


                        # set x-axis
                        plot_left.set_xlabel(
                            'epoch', 
                        )
                        plot_right.set_xlabel(
                            'epoch', 
                        )
                        
                        # print results
                        print(profile_type)
                        print(exp_type)
                        
                        # set layout tight
                        fig.tight_layout()
                        
                        # save figure
                        if HYPER_VIS.SAVE_RESULTS:
                            saving_path = path_to_figures + 'main_experiments.pdf'
                            fig.savefig(saving_path)

                       
def plot_subsampling_heuristics(
    HYPER_VIS
):
    
    """ Plots experimental results for different degrees of subsampling
    candidate data points.
    """
    
    mpl.rcParams.update({'font.size': HYPER_VIS.FONTSIZE})
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
                    
                    if 'heuristic_subsampling.csv' in file_type_list:
                        path_to_subsampling = path_to_values + 'heuristic_subsampling.csv'
                        subsampling_df = pd.read_csv(path_to_subsampling)
                        
                        path_to_figures = path_to_exp + 'figures/'
                        if not os.path.exists(path_to_figures):
                            os.mkdir(path_to_figures)
                            
                        col_name_train = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'train'
                        )
                        col_name_val = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'val'
                        )
                        
                        PL_t_iter_avg = subsampling_df[col_name_train][0]
                        budget_usage = subsampling_df[col_name_train][1]
                        sensor_usage = subsampling_df[col_name_train][2]
                        PL_loss = subsampling_df[col_name_train][4]
                        RF_loss = subsampling_df[col_name_train][5]
                        PL_accuracy = 1 - min(1, PL_loss /RF_loss)
                        PL_train = subsampling_df[col_name_train][8:].dropna().values
                        PL_val = subsampling_df[col_name_val][8:].dropna().values
                        
                        legend_PL = 'PDL: baseline  1x comp  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                            budget_usage, 
                            sensor_usage,
                            PL_accuracy
                        )
                        
                        
                        # create a list of available query variables and query variants
                        query_variables_act_lrn = set()
                        query_variants_act_lrn = set()
                        
                        df_columns_list = subsampling_df.columns
                        for i in range(3, len(df_columns_list)-1, 2):
                            column_train = df_columns_list.values[i]
                            
                            for AL_variable in HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN:
                                if AL_variable in column_train:
                                    query_variables_act_lrn = query_variables_act_lrn.union({AL_variable})
                                    break
                            
                            for AL_variant in HYPER_VIS.POSSIBLE_QUERY_VARIANTS_ACT_LRN:
                                if AL_variant in column_train:
                                    query_variants_act_lrn = query_variants_act_lrn.union({AL_variant})
                                    break
                            
                        fig, ax = plt.subplots(
                            len(query_variables_act_lrn), 
                            2, 
                            figsize=(
                                20, 
                                len(query_variables_act_lrn) * HYPER_VIS.WIDTH_FACTOR
                            )
                        )
                        
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                        
                            plot_left.plot(
                                PL_train, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL
                            )
                            plot_right.plot(
                                PL_val, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL
                            )
                        
                        df_columns_list = subsampling_df.columns
                        for i in range(3, len(df_columns_list)-1, 2):
                            column_train = df_columns_list.values[i]
                            column_val = df_columns_list.values[i+1]
                                
                            AL_t_iter_avg = subsampling_df[column_train][0]
                            budget_usage = subsampling_df[column_train][1]
                            sensor_usage = subsampling_df[column_train][2]
                            AL_loss = subsampling_df[column_train][4]
                            AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                            AL_subsample_rate = subsampling_df[column_train][6]
                            AL_train = subsampling_df[column_train][8:].dropna().values
                            AL_val = subsampling_df[column_val][8:].dropna().values
                            
                            legend_AL = 'ADL:  {}x comp  {:.0%} cand  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                                round(AL_t_iter_avg / PL_t_iter_avg, 1),
                                AL_subsample_rate,
                                budget_usage, 
                                sensor_usage,
                                AL_accuracy
                            )
                            
                            for index_var, item in enumerate(query_variables_act_lrn):
                                if item in column_train:
                                    AL_variable = item
                                    break
                                    
                            for index_heur, item in enumerate(HYPER_VIS.CAND_SUBSAMPLE_TEST_LIST):
                                if item == AL_subsample_rate:
                                    plt_color = HYPER_VIS.HEURISTICS_COLOR_LIST[index_heur]
                                    break
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                            
                            plot_left.plot(
                                AL_train, 
                                color=plt_color,
                                label=legend_AL
                            )
                            plot_right.plot(
                                AL_val, 
                                color=plt_color,
                                label=legend_AL
                            )
                            
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            
                            if index_var == 0:
                                cols = [
                                    'Training losses \n {}'.format(AL_variable), 
                                    'Validation losses \n {}'.format(AL_variable)
                                ]
                                
                                if len(query_variables_act_lrn) == 1:
                                    top_row = ax
                                else:
                                    top_row = ax[0]
                                
                                for axes, col in zip(top_row, cols):
                                    axes.set_title(col)
                            else:
                                ax[index_var, 0].set_title(AL_variable)
                                ax[index_var, 1].set_title(AL_variable)
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                            

                            
                            
                            # set y-axis labels
                            plot_left.set_ylabel(
                                'L2 loss [kW²]', 
                            )

                            # set legend
                            plot_left.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )
                            plot_right.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )
                        
                        # set x-axis
                        plot_left.set_xlabel(
                            'epoch', 
                        )
                        plot_right.set_xlabel(
                            'epoch', 
                        )
                        
                        # print results
                        print(profile_type)
                        print(exp_type)
                        
                        # set layout tight
                        fig.tight_layout()
                        
                        # save figure
                        if HYPER_VIS.SAVE_RESULTS:
                            saving_path = path_to_figures + 'subsample_importance.pdf'
                            fig.savefig(saving_path)
                        

def plot_pointspercluster_heuristics(
    HYPER_VIS
):
    
    """ Plots experimental results for different degrees of queried points
    per cluster.
    """
    
    mpl.rcParams.update({'font.size': HYPER_VIS.FONTSIZE})
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
                    
                    if 'heuristic_pointspercluster.csv' in file_type_list:
                        path_to_pointspercluster = path_to_values + 'heuristic_pointspercluster.csv'
                        pointspercluster_df = pd.read_csv(path_to_pointspercluster)
                        
                        path_to_figures = path_to_exp + 'figures/'
                        if not os.path.exists(path_to_figures):
                            os.mkdir(path_to_figures)
                            
                        col_name_train = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'train'
                        )
                        col_name_val = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'val'
                        )
                        
                        PL_t_iter_avg = pointspercluster_df[col_name_train][0]
                        budget_usage = pointspercluster_df[col_name_train][1]
                        sensor_usage = pointspercluster_df[col_name_train][2]
                        PL_loss = pointspercluster_df[col_name_train][4]
                        RF_loss = pointspercluster_df[col_name_train][5]
                        PL_accuracy = 1 - min(1, PL_loss /RF_loss)
                        PL_train = pointspercluster_df[col_name_train][8:].dropna().values
                        PL_val = pointspercluster_df[col_name_val][8:].dropna().values
                        
                        legend_PL = 'PDL: baseline  1x comp  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                            budget_usage, 
                            sensor_usage,
                            PL_accuracy
                        )
                        
                        # create a list of available query variables and query variants
                        query_variables_act_lrn = set()
                        query_variants_act_lrn = set()
                        
                        df_columns_list = pointspercluster_df.columns
                        for i in range(3, len(df_columns_list)-1, 2):
                            column_train = df_columns_list.values[i]
                            
                            for AL_variable in HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN:
                                if AL_variable in column_train:
                                    query_variables_act_lrn = query_variables_act_lrn.union({AL_variable})
                                    break
                            
                            for AL_variant in HYPER_VIS.POSSIBLE_QUERY_VARIANTS_ACT_LRN:
                                if AL_variant in column_train:
                                    query_variants_act_lrn = query_variants_act_lrn.union({AL_variant})
                                    break
                            
                        fig, ax = plt.subplots(
                            len(query_variables_act_lrn), 
                            2, 
                            figsize=(
                                20, 
                                len(query_variables_act_lrn) * HYPER_VIS.WIDTH_FACTOR
                            )
                        )
                        
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                        
                            plot_left.plot(
                                PL_train, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL
                            )
                            plot_right.plot(
                                PL_val, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL
                            )
                        
                        df_columns_list = pointspercluster_df.columns
                        for i in range(3, len(df_columns_list)-1, 2):
                            column_train = df_columns_list.values[i]
                            column_val = df_columns_list.values[i+1]
                                
                            AL_t_iter_avg = pointspercluster_df[column_train][0]
                            budget_usage = pointspercluster_df[column_train][1]
                            sensor_usage = pointspercluster_df[column_train][2]
                            AL_loss = pointspercluster_df[column_train][4]
                            AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                            AL_cluster_rate = pointspercluster_df[column_train][7]
                            AL_train = pointspercluster_df[column_train][8:].dropna().values
                            AL_val = pointspercluster_df[column_val][8:].dropna().values
                            
                            legend_AL = 'ADL:  {}x comp  {:.0%} cluster  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                                round(AL_t_iter_avg / PL_t_iter_avg, 1),
                                AL_cluster_rate,
                                budget_usage, 
                                sensor_usage,
                                AL_accuracy
                            )
                            
                            for index_var, item in enumerate(query_variables_act_lrn):
                                if item in column_train:
                                    AL_variable = item
                                    break
                                    
                            for index_heur, item in enumerate(HYPER_VIS.POINTS_PERCLUSTER_TEST_LIST):
                                if item == AL_cluster_rate:
                                    plt_color = HYPER_VIS.HEURISTICS_COLOR_LIST[index_heur]
                                    break
                            
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                            
                            
                            plot_left.plot(
                                AL_train, 
                                color=plt_color,
                                label=legend_AL
                            )
                            plot_right.plot(
                                AL_val, 
                                color=plt_color,
                                label=legend_AL
                            )
                            
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            
                            if index_var == 0:
                                cols = [
                                    'Training losses \n {}'.format(AL_variable), 
                                    'Validation losses \n {}'.format(AL_variable)
                                ]
                                
                                if len(query_variables_act_lrn) == 1:
                                    top_row = ax
                                else:
                                    top_row = ax[0]
                                
                                for axes, col in zip(top_row, cols):
                                    axes.set_title(col)
                            else:
                                ax[index_var, 0].set_title(AL_variable)
                                ax[index_var, 1].set_title(AL_variable)
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                            
                            
                            # set y-axis labels
                            plot_left.set_ylabel(
                                'L2 loss [kW²]', 
                            )
                            
                            # set legend
                            plot_left.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )
                            plot_right.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )

                        # set x-axis
                        plot_left.set_xlabel(
                            'epoch', 
                        )
                        plot_right.set_xlabel(
                            'epoch', 
                        )

                        # print results
                        print(profile_type)
                        print(exp_type)
                        
                        # set layout tight
                        fig.tight_layout()
                        
                        # save figure
                        if HYPER_VIS.SAVE_RESULTS:
                            saving_path = path_to_figures + 'pointspercluster_importance.pdf'
                            fig.savefig(saving_path)

                     
def plot_querybycoordinate_heuristics(
    HYPER_VIS
):
    
    """ Plots experimental results for querying data points by
    embedded coordinates in time and space.
    """
    
    mpl.rcParams.update({'font.size': HYPER_VIS.FONTSIZE})
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
                    
                    if 'heuristic_querybycoordinate.csv' in file_type_list:
                        path_to_querybycoordinate = path_to_values + 'heuristic_querybycoordinate.csv'
                        querybycoordinate_df = pd.read_csv(path_to_querybycoordinate)
                        
                        path_to_figures = path_to_exp + 'figures/'
                        if not os.path.exists(path_to_figures):
                            os.mkdir(path_to_figures)
                            
                        col_name_train = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'train'
                        )
                        col_name_val = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'val'
                        )
                        
                        PL_t_iter_avg = querybycoordinate_df[col_name_train][0]
                        budget_usage = querybycoordinate_df[col_name_train][1]
                        sensor_usage = querybycoordinate_df[col_name_train][2]
                        PL_loss = querybycoordinate_df[col_name_train][4]
                        RF_loss = querybycoordinate_df[col_name_train][5]
                        PL_accuracy = 1 - min(1, PL_loss /RF_loss)
                        PL_train = querybycoordinate_df[col_name_train][8:].dropna().values
                        PL_val = querybycoordinate_df[col_name_val][8:].dropna().values
                        
                        legend_PL = 'PDL: baseline  1x comp  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                            budget_usage, 
                            sensor_usage,
                            PL_accuracy
                        )
                        
                        # create a list of available query variables and query variants
                        query_variables_act_lrn = set()
                        query_variants_act_lrn = set()
                        
                        df_columns_list = querybycoordinate_df.columns
                        for i in range(3, len(df_columns_list)-1, 2):
                            column_train = df_columns_list.values[i]
                            
                            for AL_variable in HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN:
                                if AL_variable in column_train:
                                    query_variables_act_lrn = query_variables_act_lrn.union({AL_variable})
                                    break
                            
                            for AL_variant in HYPER_VIS.POSSIBLE_QUERY_VARIANTS_ACT_LRN:
                                if AL_variant in column_train:
                                    query_variants_act_lrn = query_variants_act_lrn.union({AL_variant})
                                    break
                            
                        fig, ax = plt.subplots(
                            len(query_variables_act_lrn), 
                            2, 
                            figsize=(
                                20, 
                                len(query_variables_act_lrn) * HYPER_VIS.WIDTH_FACTOR
                            )
                        )
                        
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                        
                            plot_left.plot(
                                PL_train, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL
                            )
                            plot_right.plot(
                                PL_val, 
                                color='b', 
                                linestyle='--', 
                                label=legend_PL
                            )    
                        df_columns_list = querybycoordinate_df.columns
                        for i in range(3, len(df_columns_list)-1, 2):
                            column_train = df_columns_list.values[i]
                            column_val = df_columns_list.values[i+1]
                                
                            AL_t_iter_avg = querybycoordinate_df[column_train][0]
                            budget_usage = querybycoordinate_df[column_train][1]
                            sensor_usage = querybycoordinate_df[column_train][2]
                            AL_loss = querybycoordinate_df[column_train][4]
                            AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                            AL_subsample_rate = querybycoordinate_df[column_train][6]
                            AL_train = querybycoordinate_df[column_train][8:].dropna().values
                            AL_val = querybycoordinate_df[column_val][8:].dropna().values
                            
                            legend_AL = 'ADL:  {}x comp  {:.0%} cand  {:.0%} data  {:.0%} sensors {:.0%} accuracy'.format(
                                round(AL_t_iter_avg / PL_t_iter_avg, 1),
                                AL_subsample_rate,
                                budget_usage, 
                                sensor_usage,
                                AL_accuracy
                            )
                            
                            for index_var, item in enumerate(query_variables_act_lrn):
                                if item in column_train:
                                    AL_variable = item
                                    break
                            
                            for index_heur, item in enumerate(HYPER_VIS.CAND_SUBSAMPLE_TEST_LIST):
                                if item == AL_subsample_rate:
                                    plt_color = HYPER_VIS.HEURISTICS_COLOR_LIST[index_heur]
                                    break
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                            
                            plot_left.plot(
                                AL_train, 
                                color=plt_color,
                                label=legend_AL
                            )
                            plot_right.plot(
                                AL_val, 
                                color=plt_color,
                                label=legend_AL
                            )
                            
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            
                            if index_var == 0:
                                cols = [
                                    'Training losses \n {}'.format(AL_variable), 
                                    'Validation losses \n {}'.format(AL_variable)
                                ]
                                
                                if len(query_variables_act_lrn) == 1:
                                    top_row = ax
                                else:
                                    top_row = ax[0]
                                
                                for axes, col in zip(top_row, cols):
                                    axes.set_title(col)
                            else:
                                ax[index_var, 0].set_title(AL_variable)
                                ax[index_var, 1].set_title(AL_variable)
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                            
                            # set y-axis labels
                            plot_left.set_ylabel(
                                'L2 loss [kW²]', 
                            )

                            # set legend
                            plot_left.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )
                            plot_right.legend(
                                loc='best', 
                                frameon=False,
                                fontsize=HYPER_VIS.LEGEND_FONTSIZE
                            )
                        
                        # set x-axis
                        plot_left.set_xlabel(
                            'epoch', 
                        )
                        plot_right.set_xlabel(
                            'epoch', 
                        )
                        
                        # print results
                        print(profile_type)
                        print(exp_type)
                        
                        # set layout tight
                        fig.tight_layout()
                        
                        # save figure
                        if HYPER_VIS.SAVE_RESULTS:
                            saving_path = path_to_figures + 'querybycoordinate_importance.pdf'
                            fig.savefig(saving_path)
                        
                        
def plot_sequence_importance(
    HYPER_VIS
):
    
    """ Plots experimental results for querying data points by
    random ADL sequence.
    """
    
    # create list of custom lines for custom legend
    custom_lines = [
        Line2D([0], [0], color='b', linestyle="--"),
        Line2D([0], [0], color='b')
    ]
    mpl.rcParams.update({'font.size': HYPER_VIS.FONTSIZE})
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
                    
                    if 'sequence_importance.csv' in file_type_list:
                        path_to_seqimportance = path_to_values + 'sequence_importance.csv'
                        seqimportance_df = pd.read_csv(path_to_seqimportance)
                        
                        path_to_figures = path_to_exp + 'figures/'
                        if not os.path.exists(path_to_figures):
                            os.mkdir(path_to_figures)
                            
                        # create a list of available query variables and query variants
                        query_variables_act_lrn = set()
                        query_variants_act_lrn = set()
                        
                        df_columns_list = seqimportance_df.columns
                        for i in range(3, len(df_columns_list)-1, 2):
                            column_train = df_columns_list.values[i]
                            
                            for AL_variable in HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN:
                                if AL_variable in column_train:
                                    query_variables_act_lrn = query_variables_act_lrn.union({AL_variable})
                                    break
                            
                            for AL_variant in HYPER_VIS.POSSIBLE_QUERY_VARIANTS_ACT_LRN:
                                if AL_variant in column_train:
                                    query_variants_act_lrn = query_variants_act_lrn.union({AL_variant})
                                    break
                            
                        fig, ax = plt.subplots(
                            len(query_variables_act_lrn), 
                            2, 
                            figsize=(
                                20, 
                                len(query_variables_act_lrn) * HYPER_VIS.WIDTH_FACTOR
                            )
                        )
                        
                        df_columns_list = seqimportance_df.columns
                        for i in range(1, len(df_columns_list)-1, 4):
                            
                            column_train = df_columns_list.values[i]
                            column_val = df_columns_list.values[i+1]
                            column_train_random = df_columns_list.values[i+2]
                            column_val_random = df_columns_list.values[i+3]
                            
                            AL_train = seqimportance_df[column_train][3:].dropna().values
                            AL_val = seqimportance_df[column_val][3:].dropna().values
                            AL_train_random = seqimportance_df[column_train_random][3:].dropna().values
                            AL_val_random = seqimportance_df[column_val_random][3:].dropna().values
                            
                            for index_var, item in enumerate(query_variables_act_lrn):
                                if item in column_train:
                                    AL_variable = item
                                    break
                                    
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]
                                
                            plot_left.plot(
                                AL_train
                            )
                            plot_left.plot(
                                AL_train_random,
                                linestyle="--"
                            )
                            plot_right.plot(
                                AL_val
                            )
                            plot_right.plot(
                                AL_val_random,
                                linestyle="--"
                            )
                            
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):

                            if index_var == 0:
                                cols = [
                                    'Training losses \n {}'.format(AL_variable), 
                                    'Validation losses \n {}'.format(AL_variable)
                                ]
                                
                                if len(query_variables_act_lrn) == 1:
                                    top_row = ax
                                else:
                                    top_row = ax[0]
                                
                                for axes, col in zip(top_row, cols):
                                    axes.set_title(col)
                            else:
                                ax[index_var, 0].set_title(AL_variable)
                                ax[index_var, 1].set_title(AL_variable)
                            
                            if len(query_variables_act_lrn) == 1:
                                plot_left = ax[0]
                                plot_right = ax[1]
                            else:
                                plot_left = ax[index_var, 0]
                                plot_right = ax[index_var, 1]

                            # set legend
                            plot_left.legend(
                                custom_lines, 
                                ['random sequence', 'original sequence'], 
                                loc="best", 
                                frameon=False
                            )
                            plot_right.legend(
                                custom_lines, 
                                ['random sequence', 'original sequence'], 
                                loc="best", 
                                frameon=False
                            )

                            # set y-axis labels
                            plot_left.set_ylabel(
                                'L2 loss [kW²]', 
                            )


                        # set x-axis
                        plot_left.set_xlabel(
                            'epoch', 
                        )
                        plot_right.set_xlabel(
                            'epoch', 
                        )
                        
                        # print results
                        print(profile_type)
                        print(exp_type)

                        # set layout tight
                        fig.tight_layout()

                        # save figure
                        if HYPER_VIS.SAVE_RESULTS:
                            saving_path = path_to_figures + 'sequence_importance.pdf'
                            fig.savefig(saving_path)
                        
def plot_results_summary(
    HYPER_VIS,
    result_type='result_summary'
):
    """ Plots a results or heuristics summary. For heuristics summary,
    pass the argument result_type="heuristic_summary".
    """
    
    mpl.rcParams.update({'font.size': HYPER_VIS.FONTSIZE})
    custom_lines = [
        Line2D([0], [0], color='b', linestyle="--"),
        Line2D([0], [0], color='b')
    ]
    if result_type=='result_summary':
        meta_dict = HYPER_VIS.RESULT_SUMMARY
    elif result_type=='heuristic_summary':
        meta_dict = HYPER_VIS.HEURISTIC_SUMMARY
    
    fig, ax = plt.subplots(
        meta_dict['n_rows'], 
        meta_dict['n_cols'], 
        figsize=(
            20, 
            meta_dict['n_rows'] * HYPER_VIS.WIDTH_FACTOR
        )
    )
    
    plot_counter = 0
    for plot_item in meta_dict['plot_list']:
        
        # set subplot titles
        ax[plot_item['row'], plot_item['col']].set_title(
            HYPER_VIS.SUB_TITLE_LIST[plot_counter]
        )
        
        # increment plot counter for subtitle list increment
        plot_counter += 1
        
        if plot_item['exp_choice'] == 'main_experiments':
            file_name = 'results.csv'
        elif plot_item['exp_choice'] == 'subsample_importance':
            file_name = 'heuristic_subsampling.csv'
        elif plot_item['exp_choice'] == 'pointspercluster_importance':
            file_name = 'heuristic_pointspercluster.csv'
        elif plot_item['exp_choice'] == 'querybycoordinate_importance':
            file_name = 'heuristic_querybycoordinate.csv'
        elif plot_item['exp_choice'] == 'sequence_importance':
            file_name = 'sequence_importance.csv'
           
        path_to_file = (
            HYPER_VIS.PATH_TO_RESULTS 
            + plot_item['profile_type'] + '/'
            + plot_item['pred_type'] + '/'
            + plot_item['exp_type'] + '/values/'
            + file_name
        )
        
        results_df = pd.read_csv(path_to_file)
        
        if plot_item['exp_choice'] != 'sequence_importance':
            col_name_train = (
                plot_item['pred_type'] 
                + ' None ' 
                + 'PL ' 
                + 'train'
            )
            col_name_val = (
                plot_item['pred_type'] 
                + ' None ' 
                + 'PL ' 
                + 'val'
            )

            PL_t_iter_avg = results_df[col_name_train][0]
            budget_usage = results_df[col_name_train][1]
            sensor_usage = results_df[col_name_train][2]
            PL_loss = results_df[col_name_train][4]
            RF_loss = results_df[col_name_train][5]
            PL_accuracy = 1 - min(1, PL_loss /RF_loss)
            PL_train = results_df[col_name_train][8:].dropna().values
            PL_val = results_df[col_name_val][8:].dropna().values

            legend_PL_train = 'PDL baseline:  1x comp'
            
            if result_type=='result_summary':
                legend_PL_val = 'PDL baseline:  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                    budget_usage, 
                    sensor_usage,
                    PL_accuracy
                )
            elif result_type=='heuristic_summary':
                legend_PL_val = 'PDL baseline:  1x comp  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                    budget_usage, 
                    sensor_usage,
                    PL_accuracy
                )
            
            if plot_item['plot_type'] == 'train':
                PL_plot = PL_train
                PL_legend = legend_PL_train
            elif plot_item['plot_type'] == 'val':
                PL_plot = PL_val
                PL_legend = legend_PL_val
                
            if plot_item['exp_choice'] != 'main_experiments':
                PL_legend = legend_PL_val
        
            ax[plot_item['row'], plot_item['col']].plot(
                PL_plot, 
                color='b', 
                linestyle='--', 
                label=PL_legend
            )
            
        if plot_item['exp_choice'] == 'main_experiments':                

            path_to_hyper = (
                HYPER_VIS.PATH_TO_RESULTS 
                + plot_item['profile_type'] + '/'
                + plot_item['pred_type'] + '/'
                + plot_item['exp_type'] + '/values/'
                + 'hyper.csv'
            )
            hyper_df = pd.read_csv(path_to_hyper)
                
            query_variables_act_lrn = hyper_df['query_variables_act_lrn'].dropna()
            query_variants_act_lrn = hyper_df['query_variants_act_lrn'].dropna()
        
            for index_var, AL_variable in enumerate(query_variables_act_lrn):
                
                if AL_variable == plot_item['AL_variable']:
                    
                    for index_method, AL_variant in enumerate(query_variants_act_lrn):
                        col_name_train = (
                            plot_item['pred_type'] 
                            + ' ' 
                            + AL_variable 
                            + ' ' 
                            + AL_variant 
                            + ' train'
                        )
                        col_name_val = (
                            plot_item['pred_type'] 
                            + ' ' 
                            + AL_variable 
                            + ' ' 
                            + AL_variant 
                            + ' val'
                        )
                        
                        # get training losses for mode 1 with validation updates
                        AL_t_iter_avg = results_df[col_name_train][0]
                        budget_usage = results_df[col_name_train][1]
                        sensor_usage = results_df[col_name_train][2]
                        AL_loss = results_df[col_name_train][4]
                        RF_loss = results_df[col_name_train][5]
                        AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                        AL_train = results_df[col_name_train][8:].dropna().values
                        AL_val = results_df[col_name_val][8:].dropna().values

                        # create the legends
                        legend_train = 'ADL {}:  {}x comp'.format(
                            AL_variant, 
                            round(AL_t_iter_avg / PL_t_iter_avg, 1)
                        )
                        legend_val = 'ADL {}:  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                            AL_variant, 
                            budget_usage, 
                            sensor_usage,
                            AL_accuracy
                        )
                        
                        if plot_item['plot_type'] == 'train':
                            AL_plot = AL_train
                            AL_legend = legend_train
                        elif plot_item['plot_type'] == 'val':
                            AL_plot = AL_val
                            AL_legend = legend_val
                            
                        ax[plot_item['row'], plot_item['col']].plot(
                            AL_plot, 
                            label=AL_legend
                        )
                            
            
        elif plot_item['exp_choice'] == 'subsample_importance':
            
            df_columns_list = results_df.columns
            for i in range(3, len(df_columns_list)-1, 2):
                column_train = df_columns_list.values[i]
                column_val = df_columns_list.values[i+1]

                AL_t_iter_avg = results_df[column_train][0]
                budget_usage = results_df[column_train][1]
                sensor_usage = results_df[column_train][2]
                AL_loss = results_df[column_train][4]
                AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                AL_subsample_rate = results_df[column_train][6]
                AL_train = results_df[column_train][8:].dropna().values
                AL_val = results_df[column_val][8:].dropna().values
                
                if plot_item['plot_type'] == 'train':
                    AL_plot = AL_train
                elif plot_item['plot_type'] == 'val':
                    AL_plot = AL_val
                
                legend_AL = 'ADL:  {}x comp  {:.0%} cand  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                    round(AL_t_iter_avg / PL_t_iter_avg, 1),
                    AL_subsample_rate,
                    budget_usage, 
                    sensor_usage,
                    AL_accuracy
                )

                for index_var, item in enumerate(HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN):
                    if item in column_train:
                        AL_variable = item
                        break

                for index_heur, item in enumerate(HYPER_VIS.CAND_SUBSAMPLE_TEST_LIST):
                    if item == AL_subsample_rate:
                        plt_color = HYPER_VIS.HEURISTICS_COLOR_LIST[index_heur]
                        break
                
                if AL_variable == plot_item['AL_variable']:
                    ax[plot_item['row'], plot_item['col']].plot(
                        AL_plot, 
                        color=plt_color, 
                        label=legend_AL
                    )
                 
            
        elif plot_item['exp_choice'] == 'pointspercluster_importance':
            
            df_columns_list = results_df.columns
            for i in range(3, len(df_columns_list)-1, 2):
                column_train = df_columns_list.values[i]
                column_val = df_columns_list.values[i+1]

                AL_t_iter_avg = results_df[column_train][0]
                budget_usage = results_df[column_train][1]
                sensor_usage = results_df[column_train][2]
                AL_loss = results_df[column_train][4]
                AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                AL_cluster_rate = results_df[column_train][7]
                AL_train = results_df[column_train][8:].dropna().values
                AL_val = results_df[column_val][8:].dropna().values
                
                if plot_item['plot_type'] == 'train':
                    AL_plot = AL_train
                elif plot_item['plot_type'] == 'val':
                    AL_plot = AL_val
                
                legend_AL = 'ADL:  {}x comp  {:.0%} cluster  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                    round(AL_t_iter_avg / PL_t_iter_avg, 1),
                    AL_cluster_rate,
                    budget_usage, 
                    sensor_usage,
                    AL_accuracy
                )

                for index_var, item in enumerate(HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN):
                    if item in column_train:
                        AL_variable = item
                        break

                for index_heur, item in enumerate(HYPER_VIS.POINTS_PERCLUSTER_TEST_LIST):
                    if item == AL_cluster_rate:
                        plt_color = HYPER_VIS.HEURISTICS_COLOR_LIST[index_heur]
                        break
                
                if AL_variable == plot_item['AL_variable']:
                    ax[plot_item['row'], plot_item['col']].plot(
                        AL_plot, 
                        color=plt_color, 
                        label=legend_AL
                    )
        
        
        elif plot_item['exp_choice'] == 'querybycoordinate_importance':
            
            df_columns_list = results_df.columns
            for i in range(3, len(df_columns_list)-1, 2):
                column_train = df_columns_list.values[i]
                column_val = df_columns_list.values[i+1]

                AL_t_iter_avg = results_df[column_train][0]
                budget_usage = results_df[column_train][1]
                sensor_usage = results_df[column_train][2]
                AL_loss = results_df[column_train][4]
                AL_accuracy = 1 - min(1, AL_loss/RF_loss)
                AL_subsample_rate = results_df[column_train][6]
                AL_train = results_df[column_train][8:].dropna().values
                AL_val = results_df[column_val][8:].dropna().values
                
                if plot_item['plot_type'] == 'train':
                    AL_plot = AL_train
                elif plot_item['plot_type'] == 'val':
                    AL_plot = AL_val
                
                legend_AL = 'ADL:  {}x comp  {:.0%} cand  {:.0%} data  {:.0%} sensors  {:.0%} accuracy'.format(
                    round(AL_t_iter_avg / PL_t_iter_avg, 1),
                    AL_subsample_rate,
                    budget_usage, 
                    sensor_usage,
                    AL_accuracy
                )

                for index_var, item in enumerate(HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN):
                    if item in column_train:
                        AL_variable = item
                        break

                for index_heur, item in enumerate(HYPER_VIS.CAND_SUBSAMPLE_TEST_LIST):
                    if item == AL_subsample_rate:
                        plt_color = HYPER_VIS.HEURISTICS_COLOR_LIST[index_heur]
                        break
                
                if AL_variable == plot_item['AL_variable']:
                    ax[plot_item['row'], plot_item['col']].plot(
                        AL_plot, 
                        color=plt_color, 
                        label=legend_AL
                    )
                    
                    
        elif plot_item['exp_choice'] == 'sequence_importance':
            
            df_columns_list = results_df.columns
            for i in range(1, len(df_columns_list)-1, 4):

                column_train = df_columns_list.values[i]
                column_val = df_columns_list.values[i+1]
                column_train_random = df_columns_list.values[i+2]
                column_val_random = df_columns_list.values[i+3]

                AL_train = results_df[column_train][3:].dropna().values
                AL_val = results_df[column_val][3:].dropna().values
                AL_train_random = results_df[column_train_random][3:].dropna().values
                AL_val_random = results_df[column_val_random][3:].dropna().values
                
                for index_var, item in enumerate(HYPER_VIS.POSSIBLE_QUERY_VARIABLES_ACT_LRN):
                    if item in column_train:
                        AL_variable = item
                        break
                
                if AL_variable == plot_item['AL_variable']:
                    
                    if plot_item['plot_type'] == 'train':
                        AL_plot = AL_train
                        AL_plot_random = AL_train_random
                    elif plot_item['plot_type'] == 'val':
                        AL_plot = AL_val
                        AL_plot_random = AL_val_random
                        
                    ax[plot_item['row'], plot_item['col']].plot(
                        AL_plot 
                    )
                    ax[plot_item['row'], plot_item['col']].plot(
                        AL_plot_random,
                        linestyle='--'
                    )
                    
        
        # set subplot legend
        if plot_item['exp_choice'] == 'sequence_importance':
            ax[plot_item['row'], plot_item['col']].legend(
                  custom_lines, 
                  ['random sequence', 'original sequence'], 
                  loc="best", 
                  frameon=False,
                  fontsize=HYPER_VIS.LEGEND_FONTSIZE
            )
        else:
            ax[plot_item['row'], plot_item['col']].legend(
                loc='best', 
                frameon=False,
                fontsize=HYPER_VIS.LEGEND_FONTSIZE
            )
        
        # yet subplot axis limits
        ax[plot_item['row'], plot_item['col']].set_ylim(plot_item['y_lims'])
    
    
    # set figure y-axis label
    for row in range(meta_dict['n_rows']):
        ax[row, 0].set_ylabel(
            'L2 loss [kW²]', 
        )
    
    # set figure x-axis label
    ax[-1, 0].set_xlabel(
        'epoch', 
    )
    ax[-1, 1].set_xlabel(
        'epoch', 
    )
    
    # set layout tight
    fig.tight_layout()
    if not os.path.exists(HYPER_VIS.PATH_TO_IMAGES):
        os.mkdir(HYPER_VIS.PATH_TO_IMAGES)
    
    # save 
    if HYPER_VIS.SAVE_RESULTS:
        saving_path = HYPER_VIS.PATH_TO_IMAGES + result_type + '.pdf'
        fig.savefig(saving_path)
        
        
        

