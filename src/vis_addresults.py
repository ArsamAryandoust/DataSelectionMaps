import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

class HyperParameterAdditionalVisualizing:

    """ Keeps hyper parameters together for visualizing results
    """
    
    SAVE_RESULTS = True
    PATH_TO_RESULTS = '../results/'
    
    # define for space-time selection maps
    N_ITER_PLOT = 10
    N_MESHES_SURFACE = 10
    
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
    
                
    profile_type_list = os.listdir(HYPER_ADDVIS.PATH_TO_RESULTS)
    counter = 0
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
    
    profile_type_list = os.listdir(HYPER_ADDVIS.PATH_TO_RESULTS)
    counter = 0
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

                # skip cases where we validate against queried data too
                if valup == 0:
                    # increment result index counter
                    result_index_counter += 1
                    continue
                
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
                            
                        path_to_saving_spacetime = path_to_figures + 'space-time selection/'
                        if not os.path.exists(path_to_saving_spacetime):
                            os.mkdir(path_to_saving_spacetime)
