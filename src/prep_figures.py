# create smart meter deployment map
import plotly.express as px
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

import pandas as pd
from matplotlib.lines import Line2D

import os



def prep_global_smart_meter_adoption(
    save_results=False
):

    # provide path to input data
    path_to_country_data = '../data/public/meter deployment country handfilled.csv'

    # provide the path to saving figure
    path_to_folder = '../images/global meter deployment/'

    if not os.path.exists(path_to_folder):
        os.mkdir(path_to_folder)


    # list of themese at https://plotly.com/python/templates/
    template_list =[
    #    "plotly", 
    #    "plotly_white", 
        "plotly_dark", 
    #    "ggplot2", 
        "seaborn", 
    #    "simple_white", 
    #    "none"
    ]

    # list of colors at https://plotly.com/python/builtin-colorscales/
    choropleth_color_list = [
        px.colors.diverging.BrBG,
        px.colors.diverging.PRGn,
        px.colors.diverging.PiYG,
        px.colors.diverging.PiYG,
        px.colors.diverging.PuOr,
        px.colors.diverging.RdBu,
        px.colors.diverging.RdYlGn,
        px.colors.diverging.Spectral,
        px.colors.diverging.balance,
        px.colors.diverging.delta,
        px.colors.diverging.curl,
        px.colors.diverging.oxy,
    ]

    # get the data
    df = pd.read_csv(path_to_country_data)

    # iterate over theme template list for figure
    for template in template_list:
        
        # iterate over choropleth color list
        for choro_index, choropleth_color in enumerate(choropleth_color_list):
            
            fig = px.choropleth(
                df, 
                locations='iso_alpha',
                color='deployment',
                hover_name='country',
                color_continuous_scale=choropleth_color, 
                template=template,
                width=1200, 
                height=800
            )

            fig.update_layout(
                coloraxis_colorbar=dict(
                    title='adoption [%]',
                    thicknessmode="pixels", 
                    thickness=14,
                    len=0.8,
                )
            )
            fig.update_geos(visible=False)
            
            # save results
            if save_results:
                figure_name = template + '_'+ str(choro_index) + '.pdf'
                path_to_saving_figure = path_to_folder + figure_name
                fig.write_image(path_to_saving_figure)
            
    fig.show()

    # choose a font size
    FONTSIZE = 20

    continent_list = list(set(df['continent']))
    for continent in continent_list:
        continent_df = df[df['continent']==continent]
        fig, axs = plt.subplots(figsize=(14, 5.5)) 
        ax = continent_df['deployment'].plot.hist(ax=axs, bins=30, rwidth=0.5, fontsize=FONTSIZE, width=1)
        ax.set_ylabel('number countries', fontsize=FONTSIZE)
        ax.set_xlabel('smart meter adoption [%]', fontsize=FONTSIZE)
        ax.set_xlim([-5, 105])
        ax.set_title(continent, fontsize=FONTSIZE)
        
        # save results
        if save_results:
            saving_path = path_to_folder + '0_hist_' + continent + '.pdf'
            fig.savefig(saving_path)
        

    # hand select one configuration for more scoped sub-maps
    continent_list = ['africa', 'asia', 'europe', 'south america']
    template = template_list[1]
    choropleth_color = choropleth_color_list[4]

    for template in template_list:
        for continent in continent_list:
            fig = px.choropleth(
                df, 
                locations='iso_alpha',
                color='deployment',
                color_continuous_scale=choropleth_color, 
                template=template,
                width=1200, 
                height=800
            )

            fig.update_layout(coloraxis_showscale=False)
            if continent == 'south america':
                fig.update_geos(
                    visible=False,
                    center_lat=-12,
                    center_lon=-110,
                    projection_scale=2
                )
            else:
                fig.update_geos(
                    visible=False,
                    scope=continent
                )

            # save results
            if save_results:
                saving_path = path_to_folder + '1_submap_' + continent + '_' + template + '.pdf'
                fig.write_image(saving_path)
                
                
def generate_method_data():

    """Crates randomly scattered data for demonstrating active learning method."""
    
    # Select some hyper parameters
    n_clusters = 3
    n_points_per_cluster = 20
    mu1, sigma1 = 1.5, 0.7
    mu2, sigma2 = -1, 0.7
    mu3, sigma3 = 4, 0.7

    # create random normal distributed arrays
    s1 = np.random.normal(mu1, sigma1, n_points_per_cluster)
    s2 = np.random.normal(mu2, sigma2, n_points_per_cluster)
    s3 = np.random.normal(mu3, sigma3, n_points_per_cluster)

    # stack array to one matrix
    a1 = np.concatenate([s1,s2,s3])
    a2 = np.concatenate([s2, s1, s1])
    X = np.column_stack([a1, a2])

    # cluster matrix entries
    kmeans = KMeans(n_clusters=n_clusters).fit(X)

    # plot stacked arrays and cluster centers
    plt.scatter(a1,a2)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='s')


    # create lists for saving the results
    furthest_points_list = []
    closest_points_list = []
    random_points_list = []

    # iterate over number of clusters
    for i in range(n_clusters):
        
        # filter members of currently iterated cluster from matrix
        c_members = X[kmeans.labels_ == i]
        
        # get currently iterated cluster center
        c_center = kmeans.cluster_centers_[i]
        c_center = c_center.reshape(1, -1) # reshape
        
        # calculate distance of cluster members to their cluster centers
        distances = -rbf_kernel(c_members, c_center)
        
        index_furthest = np.where(distances == np.amax(distances))[0][0]
        index_closest = np.where(distances == np.amin(distances))[0][0]
        index_random = np.random.choice(n_points_per_cluster, 1)[0]
        
        furthest_point = c_members[index_furthest]
        closest_point = c_members[index_closest]
        random_point = c_members[index_random]
        
        furthest_points_list.append(furthest_point)
        closest_points_list.append(closest_point)
        random_points_list.append(random_point)
        
        
    return a1, a2, kmeans, furthest_points_list, closest_points_list, random_points_list
    
    
def create_method_figure(
    a1, 
    a2,
    kmeans, 
    furthest_points_list, 
    closest_points_list, 
    random_points_list,
    save_results=False
):

    """Takes generated data and shows plot of active learning method."""
    
    # choose a font size
    FONTSIZE = 14

    # provide the path to saving figure
    path_to_folder = '../images/manuscript/'

    if not os.path.exists(path_to_folder):
        os.mkdir(path_to_folder)

    path_to_saving_figure = path_to_folder + 'ADL_variants.pdf'

    # set the fontsize
    matplotlib.rcParams.update({'font.size': FONTSIZE})

    # create figure with suplots
    fig, axs = plt.subplots(2,2, figsize=(13,13))

    axs[0,0].scatter(a1, a2, c=kmeans.labels_, s=50)
    axs[0,0].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, marker='s')
    for random_point in random_points_list:
        x = random_point[0]
        y = random_point[1]
        axs[0,0].scatter(x, y, s=200, c='r', marker='X')

    axs[0,1].scatter(a1, a2, c=kmeans.labels_, s=50)
    axs[0,1].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, marker='s')
    for furthest_point in furthest_points_list:
        x = furthest_point[0]
        y = furthest_point[1]
        axs[0,1].scatter(x, y, s=200, c='r', marker='X')
        
    axs[1,0].scatter(a1, a2, c=kmeans.labels_, s=50)
    axs[1,0].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, marker='s')
    for closest_point in closest_points_list:
        x = closest_point[0]
        y = closest_point[1]
        axs[1,0].scatter(x, y, s=200, c='r', marker='X')
        
    axs[1,1].scatter(a1, a2, c=kmeans.labels_, s=50)
    axs[1,1].scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, marker='s')
    for closest_point in closest_points_list:
        x = closest_point[0]
        y = closest_point[1]
        axs[1,1].scatter(x, y, s=200, c='r', marker='X')
    for furthest_point in furthest_points_list:
        x = furthest_point[0]
        y = furthest_point[1]
        axs[1,1].scatter(x, y, s=200, c='r', marker='X')


    # set titles
    axs[0,0].set_title('a.')
    axs[0,1].set_title('b.')
    axs[1,0].set_title('c.')
    axs[1,1].set_title('d.')

    # create a legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='m', markersize=15, label='cluster center'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='embedded candidate'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='r', markersize=15, label='queried candidate')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.8,1))

    # one liner to remove all axes in all subplots
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    # save figure
    if save_results:
        fig.savefig(path_to_saving_figure)


def create_model_input_output_plots(
    profile_set,
    save_results=False
):

    # path to saving input and outputs
    path_to_prediction_model_figure_folder = '../images/model IO/'

    if not os.path.exists(path_to_prediction_model_figure_folder):
        os.mkdir(path_to_prediction_model_figure_folder)

    PARAMETER_LIST = [
        'delta0_valup0', 
        'delta0_valup1', 
        'delta1_valup0', 
        'delta1_valup1'
    ]

    pred_type = 'spatio-temporal'

    # set some paths
    path_to_results = '../results/' + profile_set + '/'

    for parameter in PARAMETER_LIST:
        path_to_PL_data = (
            path_to_results 
            + pred_type 
            + '/' 
            + parameter
            + '/samples/' 
            + 'PL_'
        )

        # load data
        path_to_file = path_to_PL_data + 'X_st.npy' 
        X_st = np.load(path_to_file)

        path_to_file = path_to_PL_data + 'Y.npy' 
        Y_pl = np.load(path_to_file)

        colors = plt.rcParams["axes.prop_cycle"]()
        c=next(colors)["color"]

        # plot figures
        fig, ax = plt.subplots(3, 3, figsize=(13,13))

        rnd_point = np.random.randint(len(X_st))
        ax[0, 0].plot(X_st[rnd_point, :, 0], color=c); c=next(colors)["color"]
        ax[0, 1].plot(X_st[rnd_point, :, 1], color=c); c=next(colors)["color"]
        ax[0, 2].plot(X_st[rnd_point, :, 2], color=c); c=next(colors)["color"]
        ax[1, 0].plot(X_st[rnd_point, :, 3], color=c); c=next(colors)["color"]
        ax[1, 1].plot(X_st[rnd_point, :, 4], color=c); c=next(colors)["color"]
        ax[1, 2].plot(X_st[rnd_point, :, 5], color=c); c=next(colors)["color"]
        ax[2, 0].plot(X_st[rnd_point, :, 6], color=c); c=next(colors)["color"]
        ax[2, 1].plot(X_st[rnd_point, :, 7], color=c); c=next(colors)["color"]
        ax[2, 2].plot(X_st[rnd_point, :, 8], color=c); c=next(colors)["color"]

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        
        if save_results:
            saving_path = path_to_prediction_model_figure_folder + 'meteo_input_' + parameter + '_' + pred_type + '.pdf'
            fig.savefig(saving_path)

        # create random array to simulate predictions
        random_array = np.random.uniform(low=-0.3, high=1.5, size=(96,))

        fig, ax = plt.subplots(figsize=(12,6))
        rnd_point = np.random.randint(len(Y_pl))
        _ = ax.plot(Y_pl[rnd_point, :], label='true')
        _ = ax.plot(Y_pl[rnd_point, :] * random_array, label='predicted')

        ax.legend(loc="upper right", fontsize=16)
        #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel("time [15 min]", fontsize=16)
        plt.ylabel("consumption [kW/kWh]", fontsize=16)
        
        
        if save_results:
            saving_path = path_to_prediction_model_figure_folder + 'pred_output_' + parameter + '_' + pred_type + '.pdf'
            fig.savefig(saving_path)  
