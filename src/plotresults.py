import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def vis_train_and_val(
    HYPER, 
    results_dict
):

    """ Plots training and validation loss histories of each method, sort 
    variable and prediction type against their passive learning benchmark 
    scenarios and the random forest baseline predictor. You can use between the 
    plotting options 'separate', 'both' and 'joint':
        1. 'separate': plots the performance of each method separately against 
        the passive learning case
        2. 'joint': plots the performance of all methods jointly against the 
        passive learning benchmark
        3. 'both': plots both cases of 'separate' and 'joint'
    """

    # choose the colormap
    cmap = plt.cm.viridis
        
    # create a list of colors, one color for each AL variant
    color_list = cmap(np.linspace(0, 0.8, len(HYPER.QUERY_VARIANTS_ACT_LRN)))
    
    n_methods = len(HYPER.QUERY_VARIANTS_ACT_LRN)
    n_vars = len(HYPER.QUERY_VARIABLES_ACT_LRN)


    # create a new figure for iterated prediction type
    fig, ax = plt.subplots(n_vars, 2, figsize=(20, 10 * n_vars))

    # get variable result list
    var_result_list = results_dict['AL_result']

    # get random results
    PL_results = results_dict['PL_result']

    # get baseline results
    RF_loss = results_dict['RF_result']

    for index_var, AL_variable in enumerate(HYPER.QUERY_VARIABLES_ACT_LRN):

        ### Plot method results for each sort variable ###
        
        # plot random forest baseline results
        ax[index_var, 1].axhline(
            RF_loss,
            color='r',
            linestyle='--',
            label='RF baseline',
        )
        
        ### Plot PL results once per method for benchmark ###
        train_hist = PL_results['train_hist']
        val_hist = PL_results['val_hist']

        legend_name = ('PL: {}s- {:.0%} budget'
        '- {:.0%} sensors- {:.0%} times- {:.2} loss').format(
            PL_results['iter_time_hist'][-1],
            PL_results['budget_usage_hist'][-1],
            PL_results['sensor_usage_hist'][-1],
            PL_results['streamtime_usage_hist'][-1],
            PL_results['test_loss'],
        )

        ax[index_var, 0].plot(
            train_hist, 
            color='b', 
            linestyle='--', 
            label=legend_name
        )
        ax[index_var, 1].plot(
            val_hist, 
            color='b', 
            linestyle='--', 
            label=legend_name
        )

        # get method_result_list of currently iterated prediction type
        method_result_list = var_result_list[AL_variable]

        for index_method, method in enumerate(HYPER.QUERY_VARIANTS_ACT_LRN):

            AL_result = method_result_list[method]

            train_hist = AL_result['train_hist']
            val_hist = AL_result['val_hist']

            legend_name = ('AL {}: {}s- {:.0%} budget- {:.0%} '
            'sensors- {:.0%} times- {:.2} loss').format(
                method,
                AL_result['iter_time_hist'][-1],
                AL_result['budget_usage_hist'][-1],
                AL_result['sensor_usage_hist'][-1],
                AL_result['streamtime_usage_hist'][-1],
                AL_result['test_loss'],
            )

            ax[index_var, 0].plot(
                train_hist, 
                color=color_list[index_method], 
                label=legend_name
            )
            ax[index_var, 1].plot(
                val_hist, 
                color=color_list[index_method], 
                label=legend_name
            )

        sub_title = (
            HYPER.PRED_TYPE_ACT_LRN 
            + ' predictions - query variable ' 
            + AL_variable
        )

        ax[index_var, 0].set_title(sub_title + ' training loss')
        ax[index_var, 1].set_title(sub_title + ' validation loss')

        ax[index_var, 0].set_ylabel('loss')
        ax[index_var, 1].set_ylabel('loss')

        ax[index_var, 0].set_xlabel('epoch')
        ax[index_var, 1].set_xlabel('epoch')

        ax[index_var, 0].legend(loc='best', frameon=False)
        ax[index_var, 1].legend(loc='best', frameon=False)
