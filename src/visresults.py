import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def vis_train_and_val(
    HYPER, 
    AL_result_list, 
    PL_result_list, 
    RF_results
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


    for index_pred, pred_type in enumerate(HYPER.PRED_LIST_ACT_LRN):

        # create a new figure for iterated prediction type
        fig, ax = plt.subplots(n_vars, 2, figsize=(20, 10 * n_vars))

        # get variable result list
        var_result_list = AL_result_list[index_pred]

        # get random results
        PL_results = PL_result_list[index_pred]

        # get baseline results
        RF_loss = RF_results[pred_type]

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
            train_loss = PL_results['train_loss']
            val_loss = PL_results['val_loss']

            legend_name = ('PL: {}s- {:.0%} budget'
            '- {:.0%} sensors- {:.0%} times- {:.2} loss').format(
                PL_results['iter_time'],
                PL_results['budget_usage'],
                PL_results['sensor_usage'],
                PL_results['streamtime_usage'],
                PL_results['test_loss'],
            )

            ax[index_var, 0].plot(
                train_loss, 
                color='b', 
                linestyle='--', 
                label=legend_name
            )
            ax[index_var, 1].plot(
                val_loss, 
                color='b', 
                linestyle='--', 
                label=legend_name
            )

            # get method_result_list of currently iterated prediction type
            method_result_list = var_result_list[index_var]

            for index_method, method in enumerate(HYPER.QUERY_VARIANTS_ACT_LRN):

                AL_result = method_result_list[index_method]

                train_loss = AL_result.train_loss
                val_loss = AL_result.val_loss

                legend_name = ('AL {}: {}s- {:.0%} budget- {:.0%} '
                'sensors- {:.0%} times- {:.2} loss').format(
                    method,
                    AL_result['iter_time'],
                    AL_result['budget_usage'],
                    AL_result['sensor_usage'],
                    AL_result['streamtime_usage'],
                    AL_result['test_loss'],
                )

                ax[index_var, 0].plot(
                    train_loss, 
                    color=color_list[index_method], 
                    label=legend_name
                )
                ax[index_var, 1].plot(
                    val_loss, 
                    color=color_list[index_method], 
                    label=legend_name
                )

            sub_title = (
                pred_type 
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


def vis_seq_importance(
    HYPER, 
    AL_result_list
):

    """ Plots the training and validation losses for AL query sequence vs. 
    a random query sequence of the same data points that were queried using AL. 
    """

    if HYPER.TEST_SEQUENCE_IMPORTANCE:
        
        cmap = plt.cm.viridis
        
        # create a list of colors, one color for each AL variant
        color_list = cmap(np.linspace(0, 0.8, len(HYPER.QUERY_VARIANTS_ACT_LRN)))
        
        # create list of custom lines for custom legend
        custom_lines = [
            Line2D([0], [0], color=cmap(0.9), linestyle='--'),
            Line2D([0], [0], color=cmap(0.9))
        ]
        
        n_methods = len(HYPER.QUERY_VARIANTS_ACT_LRN)
        n_vars = len(HYPER.QUERY_VARIABLES_ACT_LRN)

        for index_pred, pred_type in enumerate(HYPER.PRED_LIST_ACT_LRN):

            # create a new figure for iterated prediction type
            fig, ax = plt.subplots(n_vars, 2, figsize=(20, 10 * n_vars))

            # get variable result list
            var_result_list = AL_result_list[index_pred]

            for index_var, AL_variable in enumerate(HYPER.QUERY_VARIABLES_ACT_LRN):

                ### Plot method results for each sort variable ###

                # get method_result_list of currently iterated prediction type
                method_result_list = var_result_list[index_var]

                for index_method, method in enumerate(
                    HYPER.QUERY_VARIANTS_ACT_LRN
                ):

                    AL_result = method_result_list[index_method]

                    train_loss = AL_result.train_loss
                    val_loss = AL_result.val_loss

                    train_loss_rnd_sequence = AL_result.seqimportance_train_loss
                    val_loss_rnd_sequence = AL_result.seqimportance_val_loss

                    ax[index_var, 0].plot(
                        train_loss, 
                        color=color_list[index_method]
                    )
                    ax[index_var, 1].plot(
                        val_loss, 
                        color=color_list[index_method]
                    )

                    ax[index_var, 0].plot(
                        train_loss_rnd_sequence, 
                        linestyle='--', 
                        color=color_list[index_method]
                    )
                    ax[index_var, 1].plot(
                        val_loss_rnd_sequence, 
                        linestyle='--', 
                        color=color_list[index_method]
                    )

                sub_title = (
                    'query sequence importance for '               
                    + pred_type 
                    + ' predictions - query variable '
                    + AL_variable
                )

                ax[index_var, 0].set_title(sub_title + " training")
                ax[index_var, 1].set_title(sub_title + " validation")

                ax[index_var, 0].set_ylabel("loss")
                ax[index_var, 1].set_ylabel("loss")

                ax[index_var, 0].set_xlabel("epoch")
                ax[index_var, 1].set_xlabel("epoch")

                ax[index_var, 0].legend(
                    custom_lines, 
                    [
                        'AL data - random sequence', 
                        'AL data - AL sequence'
                    ], 
                    loc='best', 
                    frameon=False
                )
                ax[index_var, 1].legend(
                    custom_lines, 
                    [
                        'AL data - random sequence', 
                        'AL data - AL sequence'
                    ], 
                    loc='best', 
                    frameon=False
                )
