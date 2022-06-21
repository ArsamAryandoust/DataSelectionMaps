import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import tensorflow as tf

class HyperParameterAdditionalVisualizing:

    """ Keeps hyper parameters together for visualizing results
    """
    
    
    SAVE_RESULTS = True
    PATH_TO_RESULTS = '../results/'
    FONTSIZE=20
    N_ITER_PLOT = 10
    N_MESHES_SURFACE = 10
    N_DATAPOINTS_EXAMPLE_PRED = 5
    SUB_TITLE_LIST = [
        'a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.',
        'i.', 'j.', 'k.', 'l.', 'm.', 'n.', 'o.', 'p.',
        'q.', 'r.', 's.', 't.', 'u.', 'v.', 'w.', 'x.'
    ]
    
    
def plot_space_time_selection(HYPER_ADDVIS):

    """ """
    
    def create_bottom_plot(first_colname, df_initial_sensors):
    
        ax.set_title(first_colname)

        # get bound coordinates of all buildings
        min_lat = df_initial_sensors['building lat'].min()
        max_lat = df_initial_sensors['building lat'].max() 
        min_long = df_initial_sensors['building long'].min()
        max_long = df_initial_sensors['building long'].max()

        # create evenly sized arrays and meshed grid of lats and longs
        lat_surface = np.linspace(
            min_lat, 
            max_lat,
            num=HYPER_ADDVIS.N_MESHES_SURFACE
        )
        long_surface = np.linspace(
            min_long, 
            max_long,
            num=HYPER_ADDVIS.N_MESHES_SURFACE
        )
        long_surface, lat_surface  = np.meshgrid(long_surface, lat_surface)
        
        map_height = 0
        Z = np.full((len(lat_surface), 1), map_height)
        
        ax.scatter(
            df_initial_sensors['building long'], 
            df_initial_sensors['building lat'], 
            map_height,  
            alpha=1, marker='x', c='r', s=100
        ) 
        ax.plot_surface(
            long_surface, 
            lat_surface, 
            Z,  
            alpha=0.03
        )
        ax.set_zlim(map_height)
        
    def create_plot(time_data, df, df_new_sensors):
        
        # get bound coordinates of all buildings
        min_lat = df['building lat'].min()
        max_lat = df['building lat'].max() 
        min_long = df['building long'].min()
        max_long = df['building long'].max()

        # create evenly sized arrays and meshed grid of lats and longs
        lat_surface = np.linspace(
            min_lat, 
            max_lat,
            num=HYPER_ADDVIS.N_MESHES_SURFACE
        )
        long_surface = np.linspace(
            min_long, 
            max_long,
            num=HYPER_ADDVIS.N_MESHES_SURFACE
        )
        long_surface, lat_surface  = np.meshgrid(long_surface, lat_surface)
        
        # update max time point
        min_time_point, max_time_point = min(time_data), max(time_data)
        map_height = max_time_point/3 * (1 + 3* iteration/HYPER_ADDVIS.N_ITER_PLOT)
        shifting_time = max_time_point - map_height
        Z = np.full((len(lat_surface), 1), map_height)
        
        ax.scatter(
            df['building long'], 
            df['building lat'], 
            time_data - shifting_time, 
            c=time_data, alpha=0.7
        )
        ax.scatter(
            df_new_sensors['building long'], 
            df_new_sensors['building lat'], 
            map_height,  
            alpha=1, marker='x', c='r', s=100
        )
        ax.plot_surface(
            long_surface, 
            lat_surface, 
            Z,  
            alpha=0.03
        )
        ax.set_zlim(min_time_point - shifting_time, max_time_point)
        
        
    def customize_plot(iteration=None):
        
        # set angle
        ax.view_init(30, 103)
        
        # Get rid of the panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_zticks([])

        # Add the labels
        ax.set_xlabel('longitude' )
        ax.set_ylabel('latitude')
        ax.set_zlabel('time')

        # shift time (z) axis
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( 
            tmp_planes[2], tmp_planes[3], 
            tmp_planes[0], tmp_planes[1], 
            tmp_planes[4], tmp_planes[5]
        )
        
        # shift lat (y) axis
        ax.yaxis._PLANES = ( 
            tmp_planes[2], tmp_planes[3], 
            tmp_planes[0], tmp_planes[1], 
            tmp_planes[4], tmp_planes[5]
        )
        
        # set subplot titles
        if iteration is not None:
            ax.set_title('iteration {}'.format(iteration+1))
    
    
    mpl.rcParams.update({'font.size': HYPER_ADDVIS.FONTSIZE})            
    profile_type_list = os.listdir(HYPER_ADDVIS.PATH_TO_RESULTS)
    for profile_type in profile_type_list:
        path_to_results = HYPER_ADDVIS.PATH_TO_RESULTS + profile_type + '/'
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
                    
                    if 'spacetime_selection.csv' in file_type_list:
                        # import building meta data
                        path_to_building_meta = '../data/private/' + profile_type + '/meta/meta buildings.csv'
                        building_meta = pd.read_csv(path_to_building_meta)
                        
                        path_to_file = path_to_values + 'spacetime_selection.csv'
                        space_time_df = pd.read_csv(path_to_file)
                        
                        path_to_hyper = path_to_values + 'hyper.csv'
                        hyper_df = pd.read_csv(path_to_hyper)
                        
                        path_to_figures = path_to_exp + 'figures/'
                        if not os.path.exists(path_to_figures):
                            os.mkdir(path_to_figures)
                            
                        path_to_saving_spacetime = path_to_figures + 'space-time selection/'
                        if not os.path.exists(path_to_saving_spacetime):
                            os.mkdir(path_to_saving_spacetime)
                        
                        query_variables_act_lrn = hyper_df['query_variables_act_lrn'].dropna()
                        query_variants_act_lrn = hyper_df['query_variants_act_lrn'].dropna()
                        
                        for AL_variable in query_variables_act_lrn:
                            for AL_variant in query_variants_act_lrn:
                        
                        
                                # create figure
                                fig = plt.figure(figsize=(16, (HYPER_ADDVIS.N_ITER_PLOT+1) * 8))
                                
                                # set the fontsize for figures
                                mpl.rcParams.update({'font.size': 16})
                                
                                colname_initial_sensors = '{} - initial sensors'.format(pred_type)
                                intial_sensors = space_time_df[colname_initial_sensors]
                                df_initial_sensors = pd.DataFrame({'building ID':intial_sensors})
                                df_initial_sensors = df_initial_sensors.merge(building_meta, on='building ID', how='left')
                                plot_counter= 1
                                ax = fig.add_subplot((HYPER_ADDVIS.N_ITER_PLOT+1), 2, plot_counter, projection='3d')
                                create_bottom_plot(
                                    'Active deep learning (ADL) \n iteration 0',
                                    df_initial_sensors
                                )
                                customize_plot()
                                plot_counter += 1
                                ax = fig.add_subplot((HYPER_ADDVIS.N_ITER_PLOT+1), 2, plot_counter, projection='3d')
                                create_bottom_plot(
                                    'Passive deep learning (PDL) \n iteration 0',
                                    df_initial_sensors
                                )
                                customize_plot()
                                plot_counter += 1
                                old_senors_AL_set = set(intial_sensors)
                                old_senors_PL_set = set(intial_sensors)
                                for iteration in range(HYPER_ADDVIS.N_ITER_PLOT):

                                    # create column names
                                    colname_AL_times = '{} {} {} - iter {} time'.format(pred_type, AL_variable, AL_variant, iteration) 
                                    colname_AL_spaces = '{} {} {} - iter {} space'.format(pred_type, AL_variable, AL_variant, iteration) 
                                    colname_PL_times = '{} None PL - iter {} time'.format(pred_type, iteration)
                                    colname_PL_spaces = '{} None PL - iter {} space'.format(pred_type, iteration)
                                    
                                    ### Plot AL results on left column ###
                                    
                                    # get data
                                    space_data_AL = space_time_df[colname_AL_spaces]
                                    time_data_AL = space_time_df[colname_AL_times]
                                    space_data_PL = space_time_df[colname_PL_spaces]
                                    time_data_PL = space_time_df[colname_PL_times]

                                    # get new sensors and set old sensors
                                    new_sensors_AL_set = set(space_data_AL).union(old_senors_AL_set)
                                    new_sensors_PL_set = set(space_data_PL).union(old_senors_PL_set)
                                    new_sensors_AL_list = list(new_sensors_AL_set - old_senors_AL_set)
                                    new_sensors_PL_list = list(new_sensors_PL_set - old_senors_PL_set)
                                    old_senors_AL_set = new_sensors_AL_set
                                    old_senors_PL_set = new_sensors_PL_set
                                    
                                    # assign lat and long to building IDs
                                    df_AL = pd.DataFrame({'building ID':space_data_AL})
                                    df_AL = df_AL.merge(building_meta, on='building ID', how='left')
                                    df_PL = pd.DataFrame({'building ID':space_data_PL})
                                    df_PL = df_PL.merge(building_meta, on='building ID', how='left')

                                    df_new_sensors_AL = pd.DataFrame({'building ID':new_sensors_AL_list})
                                    df_new_sensors_AL = df_new_sensors_AL.merge(building_meta, on='building ID', how='left')
                                    df_new_sensors_PL = pd.DataFrame({'building ID':new_sensors_PL_list})
                                    df_new_sensors_PL = df_new_sensors_PL.merge(building_meta, on='building ID', how='left')

                                    # AL temporal scatter plot
                                    ax = fig.add_subplot((HYPER_ADDVIS.N_ITER_PLOT+1), 2, plot_counter, projection='3d')
                                    create_plot(time_data_AL, df_AL, df_new_sensors_AL)
                                    customize_plot(iteration)
                                    plot_counter += 1

                                    # PL temporal scatter plot
                                    ax = fig.add_subplot((HYPER_ADDVIS.N_ITER_PLOT+1), 2, plot_counter, projection='3d')
                                    create_plot(time_data_PL, df_PL, df_new_sensors_PL)
                                    customize_plot(iteration)
                                    plot_counter+= 1


                                # create saving paths 
                                saving_path = (
                                    path_to_saving_spacetime 
                                    + pred_type
                                    + ' '
                                    + exp_type
                                    + ' '
                                    + AL_variable
                                    + ' '
                                    + AL_variant
                                    + '.pdf'
                                )

                                # create a legend
                                legend_elements = [
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='query in space-time'),
                                    Line2D([0], [0], marker='X', color='w', markerfacecolor='r', markersize=15, label='new sensor in space')
                                ]

                                # set layout tight
                                fig.tight_layout()

                                fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5,0.99))

                                # save figures
                                if HYPER_ADDVIS.PATH_TO_RESULTS:
                                    fig.savefig(saving_path, bbox_inches="tight")
                                    
                                    
def plot_budget_vs_accuracy(HYPER_ADDVIS):

    """ """
    
    mpl.rcParams.update({'font.size': HYPER_ADDVIS.FONTSIZE-8})
    profile_type_list = os.listdir(HYPER_ADDVIS.PATH_TO_RESULTS)
    for profile_type in profile_type_list:
        path_to_results = HYPER_ADDVIS.PATH_TO_RESULTS + profile_type + '/'
        pred_type_list = os.listdir(path_to_results)
        
        for pred_type in pred_type_list:
            path_to_pred = path_to_results + pred_type + '/'
            exp_type_list = os.listdir(path_to_pred)
            
            for exp_type in exp_type_list:
                path_to_exp = path_to_pred + exp_type + '/'
                result_type_list = os.listdir(path_to_exp)
                
                delta = int(exp_type[5])
                valup = int(exp_type[12])
    
                if 'values' in result_type_list:
                    path_to_values = path_to_exp + 'values/'
                    file_type_list = os.listdir(path_to_values)
                    
                    if 'budget_vs_accuracy.csv' in file_type_list:
                        # import building meta data
                        path_to_building_meta = '../data/private/' + profile_type + '/meta/meta buildings.csv'
                        building_meta = pd.read_csv(path_to_building_meta)
                        
                        path_to_file = path_to_values + 'budget_vs_accuracy.csv'
                        budget_accuracy_df = pd.read_csv(path_to_file)
                        
                        path_to_hyper = path_to_values + 'hyper.csv'
                        hyper_df = pd.read_csv(path_to_hyper)
                        
                        path_to_figures = path_to_exp + 'figures/'
                        if not os.path.exists(path_to_figures):
                            os.mkdir(path_to_figures)
                            
                        query_variables_act_lrn = hyper_df['query_variables_act_lrn'].dropna()
                        query_variants_act_lrn = hyper_df['query_variants_act_lrn'].dropna()
                        
                        # create the column name for PL lossess
                        col_name_data = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'data'
                        )
                        col_name_sensors = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'sensors'
                        )
                        col_name_streamtimes = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'streamtimes'
                        )
                        col_name_accuracy = (
                            pred_type 
                            + ' None ' 
                            + 'PL ' 
                            + 'accuracy'
                        )

                        # get PL results
                        PL_data = np.append(0, budget_accuracy_df[col_name_data].values)
                        PL_accuracy = np.append(0, budget_accuracy_df[col_name_accuracy].values)
                        
                        
                        fig, ax = plt.subplots(
                            len(query_variables_act_lrn), 
                            len(query_variants_act_lrn), 
                            figsize=(
                                3 * len(query_variants_act_lrn), 
                                3 * len(query_variables_act_lrn)
                            )
                        )
                        
                        for index_var, AL_variable in enumerate(query_variables_act_lrn):
                            for index_method, AL_variant in enumerate(query_variants_act_lrn):
                                
                                # create the column name for iterated validation loss
                                col_name_data = (
                                    pred_type 
                                    + ' ' 
                                    + AL_variable 
                                    + ' ' 
                                    + AL_variant 
                                    + ' data'
                                )
                                col_name_accuracy = (
                                    pred_type 
                                    + ' ' 
                                    + AL_variable 
                                    + ' ' 
                                    + AL_variant 
                                    + ' accuracy'
                                )

                                # get training losses for mode 1 with validation updates
                                AL_data = np.append(0, budget_accuracy_df[col_name_data].values)
                                AL_accuracy = np.append(0, budget_accuracy_df[col_name_accuracy].values)
                                
                                if len(query_variables_act_lrn) == 1:
                                    plot_ax = ax[index_method]
                                else:
                                    plot_ax = ax[index_var, index_method]
                                    
                                    
                                # plot iterated training losses
                                plot_ax.plot(
                                    AL_accuracy,
                                    color='b'
                                )

                                for x,y in enumerate(AL_accuracy):

                                    # plot annotations only on every second step
                                    if (x+1)%2 == 0:
                                        plot_ax.annotate(
                                            str(AL_data[x])+'%',
                                            (x, y+5)
                                        )

                                # plot PL accuracy.
                                # Note: Moved plotting down after plotting AL, in order to have legends aligned with height of plots
                                plot_ax.plot(
                                    PL_accuracy, 
                                    color='r',
                                )

                                plot_ax.set_ylim(
                                    top=100
                                )

                                for x,y in enumerate(PL_accuracy):
                                    # plot annotations only on every second step
                                    if (x+1)%2 == 0:
                                        plot_ax.annotate(
                                            str(PL_data[x])+'%',
                                            (x, y-5)
                                        )
                                        
                                # set subplot titles
                                
                                if len(query_variables_act_lrn) == 1:
                                    # set y-axis labels
                                    ax[0].set_ylabel(
                                        '{} \n prediction accuracy'.format(AL_variable)
                                    )

                                    # set x-axis
                                    ax[index_method].set_xlabel(
                                        'data selection \n iteration'
                                    )
                                    
                                    # set column titles
                                    if index_var == 0:
                                        ax[index_method].set_title(AL_variant)
                                else:
                                    # set y-axis labels
                                    ax[index_var, 0].set_ylabel(
                                        '{} \n prediction accuracy'.format(AL_variable)
                                    )

                                    # set x-axis
                                    ax[len(query_variables_act_lrn)-1, index_method].set_xlabel(
                                        'data selection \n iteration'
                                    )
                                    
                                    # set column titles
                                    if index_var == 0:
                                        ax[0, index_method].set_title(AL_variant)
                                    
                        # create saving paths 
                        saving_path = (
                            path_to_figures 
                            + 'budget_vs_accuracy.pdf'
                        )

                        legend_elements = [
                            Line2D([0], [0], color='b', label='Active learning'),
                            Line2D([0], [0], color='r', label='Passive learning'),
                            Line2D([0], [0], color='w', label='% = budget usage')
                        ]

                        # set layout tight
                        fig.tight_layout()

                        fig.legend(
                            handles=legend_elements,
                            bbox_to_anchor=(0.9,1.14 - 0.02 *len(query_variables_act_lrn))
                        )

                        # save figures
                        if HYPER_ADDVIS.SAVE_RESULTS:
                            fig.savefig(saving_path, bbox_inches="tight")
                            
                            
                            
def plot_exemplar_predictions(HYPER_ADDVIS):

    """ """
    
    mpl.rcParams.update({'font.size': HYPER_ADDVIS.FONTSIZE})
    profile_type_list = os.listdir(HYPER_ADDVIS.PATH_TO_RESULTS)
    for profile_type in profile_type_list:
        path_to_results = HYPER_ADDVIS.PATH_TO_RESULTS + profile_type + '/'
        pred_type_list = os.listdir(path_to_results)
        
        for pred_type in pred_type_list:
            path_to_pred = path_to_results + pred_type + '/'
            exp_type_list = os.listdir(path_to_pred)
            
            for exp_type in exp_type_list:
                path_to_exp = path_to_pred + exp_type + '/'
                result_type_list = os.listdir(path_to_exp)
                
                path_to_samples = path_to_exp + 'samples/'
                path_to_models = path_to_exp + 'models/'
                path_to_values = path_to_exp + 'values/'
                path_to_figures = path_to_exp + 'figures/'
                
                if not os.path.exists(path_to_figures):
                    os.mkdir(path_to_figures)
                    
                path_to_exemplar_pred = path_to_figures + 'exemplar predictions/'
                if not os.path.exists(path_to_exemplar_pred):
                    os.mkdir(path_to_exemplar_pred)
                
                # get hyper params
                path_to_hyper = path_to_values + 'hyper.csv'
                hyper_df = pd.read_csv(path_to_hyper)
                
                # import initial model
                path_to_initial_model = path_to_models + 'initial.h5'
                initial_model =  tf.keras.models.load_model(
                    path_to_initial_model, 
                    compile=False
                )
                
                # import PL model and test samples
                path_to_PL_model = path_to_models + 'PL.h5'
                PL_model = tf.keras.models.load_model(
                    path_to_PL_model, 
                    compile=False
                )
                
                path_to_file = path_to_samples + 'PL_X_t.npy' 
                X_t = np.load(path_to_file)

                path_to_file = path_to_samples + 'PL_X_s.npy' 
                X_s = np.load(path_to_file)

                path_to_file = path_to_samples + 'PL_X_s1.npy' 
                X_s1 = np.load(path_to_file)

                path_to_file = path_to_samples + 'PL_X_st.npy' 
                X_st = np.load(path_to_file)

                path_to_file = path_to_samples + 'PL_Y.npy' 
                Y_pl = np.load(path_to_file)
                
                # make predictions
                initial_predictions = initial_model.predict(
                    [X_t, X_s1, X_st]
                )
                PL_predictions = PL_model.predict(
                    [X_t, X_s1, X_st]
                )
                
                
                query_variables_act_lrn = hyper_df['query_variables_act_lrn'].dropna()
                query_variants_act_lrn = hyper_df['query_variants_act_lrn'].dropna()
                
                # iterate over all AL variables
                for index_var, AL_variable in enumerate(query_variables_act_lrn):
                    
                    # iterate over all AL variants
                    for index_method, AL_variant in enumerate(query_variants_act_lrn):
                    
                        # create figure
                        fig, ax = plt.subplots(
                            HYPER_ADDVIS.N_DATAPOINTS_EXAMPLE_PRED, 
                            3, 
                            sharex=True , 
                            figsize=(16, HYPER_ADDVIS.N_DATAPOINTS_EXAMPLE_PRED * 4)
                        )
                        
                        
                        AL_model_filename = '{} {}.h5'.format(
                            AL_variable, 
                            AL_variant
                        )
                        AL_sample_name = '{} {} '.format(
                            AL_variable, 
                            AL_variant
                        )
                        
                        # provide paths to initial and PL models and samples
                        path_to_AL_model = path_to_models + AL_model_filename
                        path_to_AL_data = path_to_samples + AL_sample_name
                        
                        # import models and samples for AL
                        path_to_PL_model = path_to_models + 'PL.h5'
                        AL_model =  tf.keras.models.load_model(
                            path_to_AL_model, 
                            compile=False
                        )
                
                        path_to_file = path_to_AL_data + 'X_t.npy' 
                        X_t = np.load(path_to_file)

                        path_to_file = path_to_AL_data + 'X_s.npy' 
                        X_s = np.load(path_to_file)

                        path_to_file = path_to_AL_data + 'X_s1.npy' 
                        X_s1 = np.load(path_to_file)

                        path_to_file = path_to_AL_data + 'X_st.npy' 
                        X_st = np.load(path_to_file)

                        path_to_file = path_to_AL_data + 'Y.npy' 
                        Y_al = np.load(path_to_file)
                        
                        # make predictions
                        AL_predictions = AL_model.predict(
                            [X_t, X_s1, X_st]
                        )
                        
                        # plot predictions for randomly chosen points
                        rnd_index_array_initial = np.random.choice(
                            np.arange(len(Y_pl)),
                            HYPER_ADDVIS.N_DATAPOINTS_EXAMPLE_PRED
                        )
                        rnd_index_array_PL = np.random.choice(
                            np.arange(len(Y_pl)),
                            HYPER_ADDVIS.N_DATAPOINTS_EXAMPLE_PRED
                        )
                        rnd_index_array_AL = np.random.choice(
                            np.arange(len(Y_al)), 
                            HYPER_ADDVIS.N_DATAPOINTS_EXAMPLE_PRED
                        )
                        
                        title_counter = 0
                
                        # iterate over each row of figure
                        for row in range(HYPER_ADDVIS.N_DATAPOINTS_EXAMPLE_PRED):
                            
                            plot1 = ax[row, 0].plot(
                                initial_predictions[
                                    rnd_index_array_initial[
                                        row
                                    ]
                                ]
                            )
                            ax[row, 1].plot(
                                PL_predictions[
                                    rnd_index_array_PL[
                                        row
                                    ]
                                ]
                            )
                            ax[row, 2].plot(
                                AL_predictions[
                                    rnd_index_array_AL[
                                        row
                                    ]
                                ]
                            )
                            
                            plot2 = ax[row, 0].plot(
                                Y_pl[
                                    rnd_index_array_initial[
                                        row
                                    ]
                                ]
                            )
                            ax[row, 1].plot(
                                Y_pl[
                                    rnd_index_array_PL[
                                        row
                                    ]
                                ]
                            )
                            ax[row, 2].plot(
                                Y_al[
                                    rnd_index_array_AL[
                                        row
                                    ]
                                ]
                            )
                            
                            # set title
                            for col in range(3):
                                ax[row, col].set_title(HYPER_ADDVIS.SUB_TITLE_LIST[title_counter])
                                title_counter +=1
                        
                        # add a figure legend
                        fig.legend(
                            [plot1, plot2], 
                            labels=['true load profile', 'predicted load profile'], 
                            loc='upper center', 
                            bbox_to_anchor=(0.8, 1)
                        )

                        colname_list = [
                            'Initial model \n a.', 
                            'Passive learning \n b.', 
                            'Active learning \n c.'
                        ]

                        # set col names
                        for axes, colname in zip(ax[0], colname_list):
                            axes.set_title(colname)

                        # set one y- and x-axis for all sub plots
                        fig.add_subplot(111, frame_on=False)
                        plt.tick_params(
                            labelcolor="none", 
                            bottom=False, 
                            left=False
                        )
                        plt.xlabel(
                            'time [15-min]' 
                        )
                        plt.ylabel(
                            'building electric consumption [kW]'
                        )
                        
                        filename = '{} {} {}.pdf'.format(
                            pred_type, 
                            AL_variable, 
                            AL_variant, 
                        )
                        
                        saving_path = (
                            path_to_exemplar_pred 
                            + filename
                        )

                        fig.savefig(saving_path)
            
