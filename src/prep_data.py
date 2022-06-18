# Manual profile selection
import os
import pandas as pd
import math

# clustering
from sklearn.cluster import AgglomerativeClustering

# Creating building year profiles
import numpy as np

# Getting meteorological data
import requests
import json
import time

# Getting aerial imagery for clusters
import cv2
import matplotlib.pyplot as plt

# Padding images to same size
from PIL import Image

# Generating additional profiles
from sklearn import preprocessing
import random
import rasterio

    
    
def show_profiles(dataset_index):
    
    """ Prints out load profiles for manual selection."""
    
    path_to_input = '../data/private/load profiles'
    # get a list of the data file names
    dataset_list = os.listdir(path_to_input)
    
    # sort the list
    dataset_list.sort()

    # get a dataset
    dataset = dataset_list[dataset_index]

    # Get the file name and import file
    path_to_file = path_to_input + '/' + dataset
    
    # import the dataset as pandas.DataFrame object
    single_dataset = pd.read_excel(path_to_file)

    # Remove temporal information
    single_dataset.pop('MET_CODE')

    # Save the local information
    addresses_list = single_dataset[:8]

    # remove location information
    single_dataset = single_dataset[8:]

    # get the number of columns
    n_cols = len(single_dataset.columns)

    # change max entries of dataframe to be shown
    pd.set_option('display.max_columns', n_cols)

    # displays addresses alonside the metering IDs
    display(addresses_list)

    # print dataset name for easier copy-paste
    print(dataset)

    # plot the time series data for each metering code
    _ = single_dataset.plot(
        legend=True, 
        subplots=True, 
        layout=(math.ceil(n_cols/2), 2), 
        figsize=(16, n_cols)
    )
    
    
def cluster_profiles(
    profile_set,
    distance_km,
    save_results=False
):
    """ Clusters buildings that are in certain distance to each other. """
    
    # ommit warnings
    pd.options.mode.chained_assignment = None
    
    # provide the paths to the meta input data files
    path_to_meta_building = (
        '../data/private/' + profile_set + '/meta/meta buildings handfilled.csv'
    )
    path_to_meta_profile = (
        '../data/private/' + profile_set + '/meta/meta profiles handfilled.csv'
    )

    # provide the paths to the meta output data files
    path_to_meta_building_output = (
        '../data/private/' + profile_set + '/meta/meta buildings.csv'
    )
    path_to_meta_profile_output = (
        '../data/private/' + profile_set + '/meta/meta profiles.csv'
    )

    ###
    # Convert distance in km to distance in degree ###
    ###
    # get the distance from km in terms of geographic coordinates.
    # One degree latitude at lat in m is: 111132.92 - 559.82 * cos(2*lat) + 1.175 * cos(4*lat) - 0.0023 * cos(6*lat)
    # One degreee longitude at lat in m is: 111412.84 * cos(lat) - 93.5 * cos(3*lat) + 0.118 * cos(5*lat)
    distance_dg = distance_km / 111.3

    # Import data 
    meta_building = pd.read_csv(
        path_to_meta_building
    )
    meta_profile = pd.read_csv(
        path_to_meta_profile
    )


    ###
    # Cluster data ###
    ###

    # get lat and long data matrix
    X = meta_building[
        [
            "building lat", 
            "building long"
        ]
    ].values

    # define clustering algorithm without predefined number of clusters, but with the above defined member distance
    clustering_algo = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_dg
    )

    # fit the clustering algorithm
    clustering_algo.fit(X)

    # get the labels. labels start from zero, so add 1 to start them from 1. 
    cluster_id = clustering_algo.labels_ + 1 

    # add the labels to dataframe under new column 'cluster ID'
    meta_building["cluster ID"] = cluster_id


    ###
    # Determine cluster centers ###
    ###

    # get number of clusters
    n_clusters = clustering_algo.n_clusters_

    # get the entries that are relevant only for faster iterations
    meta_clustering = meta_building[
        [
            "building lat", 
            "building long", 
            "cluster ID"
        ]
    ]

    # add two new columns for saving lats and longs of cluster centers to the meta data frame
    meta_building["cluster lat"] = 0
    meta_building["cluster long"] = 0

    # iterate over each cluster ID
    for i in range(1, n_clusters+1):

        # filter rows that have same cluster ID as in current iteration
        cluster_members_list = meta_clustering[
            meta_clustering["cluster ID"] == i
        ]

        # calculate the cluster's central latitudinal coordinate
        min_lat = cluster_members_list["building lat"].min()
        max_lat = cluster_members_list["building lat"].max()
        central_lat = (max_lat + min_lat) / 2

        # calculat the cluster's central longitudinal coordinate
        min_long = cluster_members_list["building long"].min()
        max_long = cluster_members_list["building long"].max()
        central_long = (max_long + min_long) / 2

        # add the coordinates to a list
        meta_building["cluster lat"].iloc[
            cluster_members_list.index
        ] = central_lat
        meta_building["cluster long"].iloc[
            cluster_members_list.index
        ] = central_long
        
    # Create meta profile
    meta_profile = meta_profile.merge(
        meta_building[
            [
                "cluster ID", 
                "building ID"
            ]
        ]
    )

    # Save results
    if save_results:
        meta_building.to_csv(
            path_to_meta_building_output, 
            index=False, 
            header=True
        )
        meta_profile.to_csv(
            path_to_meta_profile_output, 
            index=False, 
            header=True
        )

    # plot the generated building clusters
    _ = meta_clustering.plot.scatter(
        x="building lat", 
        y="building long", 
        c="cluster ID", 
        figsize=(16, 16)
    )
    
    
def create_toydata(
    profile_set,
    save_results=False
):
    
    """ Creates randomized and annonymized toy dataset for publication """
    
    ###
    # Set parameters ###
    ###

    # provide the paths to the meta input data files
    path_to_meta_building_input = (
        '../data/private/' + profile_set + '/meta/meta buildings.csv'
    )

    # provide the paths to the meta output data files
    path_to_meta_toy_public_building_output = (
        '../data/public/' + profile_set + '/meta/meta buildings toy.csv'
    )
    
    ###
    # Process ###
    ###

    # Import data 
    meta_building = pd.read_csv(path_to_meta_building_input)

    # calculate min and max coordinates
    lat_min = meta_building['building lat'].min()
    lat_max = meta_building['building lat'].max()
    long_min = meta_building['building long'].min()
    long_max = meta_building['building long'].max()

    # create random arrays of lats and longs
    n_buildings = len(meta_building)
    lat_rnd_array = np.random.uniform(low=lat_min, high=lat_max, size=(n_buildings,))
    long_rnd_array = np.random.uniform(low=long_min, high=long_max, size=(n_buildings,))

    # create and save as new dataframe
    meta_buildings_toy_public = pd.DataFrame(
        {
            'building ID': meta_building['building ID'], 
            'building lat': lat_rnd_array,
            'building long': long_rnd_array
        }
    )
    
    if save_results:
        meta_buildings_toy_public.to_csv(
            path_to_meta_toy_public_building_output, 
            index=False, 
            header=True
        )

    # print out the bounds for downloading map file in tif format manually from geovite.ethz.ch
    print(
        'All buildings are in the coordinate bounds of \n lat:[{}, {}] \n long:[{}, {}]'.format(
            lat_min, lat_max, long_min, long_max
        )
    )
    
    
def create_building_year_profiles(
    profile_set,
    save_results=False
):
    """ Matches building year profiles into dataset files. """
    
    ###
    # Set parameters ###
    ###

    # provide path to meta profile input file
    path_to_meta_profile = '../data/private/' + profile_set + '/meta/meta profiles.csv'

    path_to_results_folder = '../data/private/' + profile_set + '/building-year profiles/original/'
    path_to_building_year_profile = 'building-year profiles.csv'

    # provide path to load profile folder
    path_to_profile_folder = '../data/private/load profiles/'

    # import profile meta data
    meta_profile = pd.read_csv(path_to_meta_profile)


    ###
    # Create building year profiles
    ###

    # we want to save building-year profiles in files separated by year. Get all available years
    year_ID_list = meta_profile["year"].drop_duplicates().values

    # iterate over each year for which profiles are available
    for year in year_ID_list:

        # filter meta data rows by year
        meta_profile_by_year = meta_profile[
            meta_profile["year"] == year
        ]

        # get the unique list of file names for this year's profiles
        file_name_list = meta_profile[
            "file name"
        ].drop_duplicates().values

        # iterate over each file that contains profiles for this year
        for file_name in file_name_list:

            # create the path to currently iterated file
            path_to_file = path_to_profile_folder + file_name + '.xls'

            # load currently iterated file
            df = pd.read_excel(path_to_file)

            # remove the unused first rows
            df = df[8:]

            # get the time stamp array
            time_stamp = df.pop("MET_CODE")

            # change the time stamp format of the meters to a format that is equal to the meteo data time stamp
            time_stamp = time_stamp.apply(
                lambda x: 
                    x[6:10] 
                    + '-' 
                    + x[3:5] 
                    + '-' 
                    + x[:2] 
                    + ' ' 
                    + x[11:16] 
                    + ':00'
            )


            ###
            # create an empty file skeleton in the wanted order
            ###

            # check if this is the first file that we opened for this year
            if file_name == file_name_list[0]:

                # get all building IDs for this year
                building_ids = meta_profile_by_year[
                    "building ID"
                ].drop_duplicates()

                # get the indices of all building IDs for this year
                indices_building_ids = meta_profile_by_year[
                    "building ID"
                ].drop_duplicates().index

                # get the corresponding cluster IDs for this year
                cluster_ids = meta_profile_by_year[
                    "cluster ID"
                ].iloc[indices_building_ids]

                # create the rows 'building ID', 'cluster ID' and 'year'
                skeleton = pd.DataFrame(
                    [
                        "building ID", 
                        "cluster ID", 
                        "year"
                    ]
                )

                # append the time stamp to the skeleton
                skeleton = pd.concat(
                    [
                        skeleton, 
                        time_stamp
                    ]
                )

                # add zero entries to prepare skeleton into the right shape
                skeleton = skeleton.reindex(
                    columns=range(
                        len(
                            cluster_ids
                        ) 
                        + 1
                    ), 
                    fill_value=0
                )

                # reset index
                skeleton = skeleton.reset_index(drop=True)

                # get the year as an appropriately sized array
                year_array = np.full(
                    cluster_ids.shape, 
                    year
                )

                # add the building IDs, cluster IDs and years to the skeleton
                skeleton.iloc[0, 1:] = building_ids.values
                skeleton.iloc[1, 1:] = cluster_ids.values
                skeleton.iloc[2, 1:] = year_array


            # get the list of metering IDs available in the currently iterated and opened file
            meter_ids_file = df.columns.values

            # get the list of metering IDs that match between meta profile of this year and the currently imported file
            meta_profile_to_import = meta_profile_by_year[
                meta_profile_by_year[
                    "metering ID"
                ].isin(
                    meter_ids_file
                )
            ]

            # iterate over all metering IDs and add them up
            for index, meter in meta_profile_to_import.iterrows():

                # get building ID of current entry
                building_id = meter[
                    "building ID"
                ]

                # get metering ID and its index for current entry
                metering_id = meter[
                    "metering ID"
                ]
                metering_id_index = skeleton.iloc[0, :][
                    skeleton.iloc[0, :] == building_id
                ].index[0]

                # get respective load profile from currently imported file
                load_profile = df[
                    metering_id
                ].reset_index(drop=True)

                # add profile to respective line in skeleton
                skeleton.iloc[
                    3:, 
                    metering_id_index
                ] += load_profile.values

        # save the results
        if save_results:
            path_to_building_year_profile = (
                path_to_results_folder 
                + str(year) 
                + ' ' 
                + path_to_building_year_profile
            )
            skeleton.to_csv(
                path_to_building_year_profile, 
                index=False, 
                header=False
            )

    # Plot the results 
    n_subplots = len(skeleton.columns)
    _ = skeleton.iloc[3:, :].plot( 
        subplots=True, 
        layout=(math.ceil(n_subplots/2), 2), 
        figsize=(16, n_subplots)
    ) 
    
    
def download_meteo_data(
    profile_set,
    save_results=False
):
    """Downloads meteorological data """
    
    ###
    # Set parameters ###
    ###

    # provide the paths to the input and output files
    path_to_meta_building = '../data/private/' + profile_set + '/meta/meta buildings.csv'
    path_to_meteo_folder = '../data/public/' + profile_set + '/meteo data/'

    # provide path to renewables ninja token
    path_to_token = '../.ninja_token'

    # provide the user specific tokens and the api base url for renewables.ninja
    api_base = "https://www.renewables.ninja/api/"

    # import token from file
    with open(path_to_token, 'r') as file:
        token = file.readline()


    ###
    # Import and prepare building meteo file
    ###

    # import input file
    cluster_coordinate_mapping = pd.read_csv(
        path_to_meta_building
    )

    # extract only entries that we need hereafter
    cluster_coordinate_mapping = cluster_coordinate_mapping[
        [
            "years", 
            "cluster ID", 
            "cluster lat", 
            "cluster long"
        ]
    ]

    # drop the duplicate lines
    cluster_coordinate_mapping = (
        cluster_coordinate_mapping.drop_duplicates()
    )


    def recursive_call():

        counter = 0
        try:

            for index, row in cluster_coordinate_mapping.iterrows():

                # get the years for which this building cluster is measured
                years = row['years']

                # if not a string already, transform it into a string
                if not isinstance(years, str):

                    years = str(years)

                # if more than one year available, split the string by delimiter '-'
                years_list = years.split('-')

                # iterate over all years of that building
                for year in years_list:

                    year = int(float(year))
                    cluster_id = int(
                        row[
                            'cluster ID'
                        ]
                    )

                    output_file_name = (
                        'meteo_' 
                        + str(cluster_id) 
                        + '_' 
                        + str(year) 
                        + '.csv'
                    )
                    existing_file_name_list = os.listdir(
                        path_to_meteo_folder
                    )

                    if output_file_name in existing_file_name_list:

                        continue

                    else:

                        # get the coordinates of the cluster
                        lat = row['cluster lat']
                        long = row['cluster long']

                        # file does not exist yet and must be requested
                        path_to_output_file = (
                            path_to_meteo_folder 
                            + output_file_name
                        )

                        ##
                        # Weather 
                        ##
                        s = requests.session()
                        s.headers = {
                            "Authorization": "Token " + token
                        }

                        url = api_base + "data/weather"
                        args = {
                            "lat": lat,
                            "lon": long,
                            "date_from": str(year) + '-01-01',
                            "date_to": str(year) + '-12-31',
                            "local_time": True,
                            "dataset": "merra2",
                            "var_t2m": True,
                            "var_prectotland": True,
                            "var_precsnoland": True,
                            "var_snomas": True,
                            "var_rhoa": True,
                            "var_swgdn": True,
                            "var_swtdn": True,
                            "var_cldtot": True,
                            "format": 'json',
                        }

                        r = s.get(url, params=args)
                        parsed_response = json.loads(r.text)
                        weather_data = pd.read_json(
                            json.dumps(
                                parsed_response['data']
                            ), 
                            orient='index'
                        )

                        ##
                        # Wind 
                        ##
                        s = requests.session()
                        s.headers = {
                            "Authorization": "Token " + token
                        }

                        url = api_base + "data/wind"
                        args = {
                            "lat": lat,
                            "lon": long,
                            "date_from": str(year) + "-01-01",
                            "date_to": str(year) + "-12-31",
                            "capacity": 1.0,
                            "height": 10,
                            "turbine": "Vestas V80 2000",
                            "format": "json",
                            "raw": True
                        }

                        r = s.get(url, params=args)
                        parsed_response = json.loads(r.text)
                        wind_data = pd.read_json(
                            json.dumps(
                                parsed_response[
                                    "data"
                                ]
                            ), 
                            orient="index"
                        )

                        wind_speed = (
                            wind_data.wind_speed.values
                        )
                        weather_data[
                            "wind_speed"
                        ] = wind_speed

                        local_time = (
                            weather_data.local_time.values
                        )
                        for i in range(len(local_time)):

                            local_time[i] = local_time[i][:-6]

                        weather_data[
                            "local_time"
                        ] = local_time

                        ###
                        # repeat each row four times to transform 1-h resolution to 15-min resolution ###
                        ###

                        expanded_df = pd.DataFrame(
                            np.repeat(
                                weather_data.values, 
                                4, 
                                axis=0
                            )
                        )
                        expanded_df.columns = weather_data.columns

                        counter_timestep = 0
                        for i in range(len(expanded_df)):

                            if counter_timestep == 0:

                                counter_timestep += 1
                                continue

                            elif counter_timestep == 1:

                                string_replacement = '15'

                            elif counter_timestep == 2:

                                string_replacement = '30'

                            elif counter_timestep == 3:

                                string_replacement = '45'
                                counter_timestep = -1

                            string_beginning = expanded_df[
                                'local_time'
                            ][i][:14]
                            string_ending = expanded_df[
                                'local_time'
                            ][i][16:]
                            string = (
                                string_beginning 
                                + string_replacement 
                                + string_ending
                            )

                            expanded_df[
                                'local_time'
                            ][i] = string
                            counter_timestep += 1

                        # Save the file
                        if save_results:
                            expanded_df.to_csv(
                                path_to_output_file, 
                                index=False, 
                                header=True
                            )

                        # Increment counter
                        counter += 1

        except:

            # print how many files we added
            print(counter, 'files were added')

            # print the error message
            print(r.text)

            # retrieve the remaining waiting time to next api call from error message
            waiting_time = r.text[71:73]
            waiting_time = int(waiting_time)

            # pause program execution for this amount of time before making new requests
            time.sleep(waiting_time)

            # call the current function again, recursively
            recursive_call()

    # Call the recursive function to send API calls until the list is empty or program is manually interrupted
    recursive_call()


    ###
    # Visualize exemplar meteo data ###
    ###

    # set exemplar meteo data fiile name
    file_name = 'meteo_1_2014.csv'

    # create the entire path to the exemplar meteo data file 
    path_to_file = path_to_meteo_folder + file_name

    # load file
    df = pd.read_csv(path_to_file)

    # set one of the columns 'local_time' as index for later search purposes
    df = df.set_index('local_time')

    _ = df.plot(
        use_index=False, 
        legend=True, 
        figsize=(16,16), 
        fontsize=14, 
        subplots=True, 
        layout=(3,3), 
        title='Exemplar meteorological data file for the year 2014'
    )    
    
    
def get_cluster_aerial_imagery(
    profile_set='profiles_100',
    save_results=False
):
    """ Downloads aerial imagery """
    
    
    
    ###
    # Set parameters ###
    ###


    # provide the path to building data
    path_to_meta_building = '../data/private/' + profile_set + '/meta/meta buildings.csv'

    # provide the path to aerial imagery cluster folder
    path_to_cluster_image_folder = '../data/private/' + profile_set + '/cluster imagery/'

    # provide the API base urls for geovite.ethz.ch
    api_base_ortho = (
        'https://geovite.ethz.ch/ServiceProxyServlet?server=24&serverpath=/cgi-bin/swissimage/wms.fcgi?'
    )
    api_base_dem = (
        'https://geovite.ethz.ch/ServiceProxyServlet?server=24&serverpath=/cgi-bin/swissalti/wms.fcgi?'
    )            

    # Choose image parameters
    image_width = 1024
    image_height = 1024

    zoom_level_lats = { 
        'zoom1' : 0.1, 
        'zoom2' : 0.05, 
        'zoom3' : 0.01
    }

    # calculate the longitudinal variance in degrees from that as
    long_var_deg_list = [
        x / math.cos(x) for x in zoom_level_lats.values()
    ] 


    # provide the set of fixed parameters 
    fixed_params_ortho = (
        'VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap&LAYERS=latest&STYLES=default&SRS=EPSG%3A4326&WIDTH=' 
        + str(image_width) 
        + '&HEIGHT=' 
        + str(image_height) 
        + '&FORMAT=image/png&'
    )
    fixed_params_dem_relief = (
        'VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap&LAYERS=swissALTI3D2018relief&STYLES=&SRS=EPSG%3A4326&WIDTH=' 
        + str(image_width) 
        + '&HEIGHT=' 
        + str(image_height) 
        + '&FORMAT=image/png&'
    )
    fixed_params_dem_aspect = (
        'VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap&LAYERS=swissALTI3D2018aspect&STYLES=&SRS=EPSG%3A4326&WIDTH=' 
        + str(image_width) 
        + '&HEIGHT=' 
        + str(image_height) 
        + '&FORMAT=image/png&'
    )
    fixed_params_dem_slope = (
        'VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap&LAYERS=swissALTI3D2018slope&STYLES=&SRS=EPSG%3A4326&WIDTH=' 
        + str(image_width) 
        + '&HEIGHT=' 
        + str(image_height) 
        + '&FORMAT=image/png&'
    )

    ###
    # Import data
    ###

    # import input file
    cluster_coordinate_mapping = pd.read_csv(
        path_to_meta_building
    )

    # extract only entries that we need hereafter
    cluster_coordinate_mapping = cluster_coordinate_mapping[
        [
            'cluster ID', 
            'cluster lat', 
            'cluster long'
        ]
    ]

    # drop the duplicate lines
    cluster_coordinate_mapping = (
        cluster_coordinate_mapping.drop_duplicates()
    )

    ###
    # Collect aerial imagery for each cluster ###
    ###

    # iterate over the list of meter clusters
    for _, row in cluster_coordinate_mapping.iterrows():

        # get the currently iterated cluster id and coordinates
        cluster_id = int(row['cluster ID'])
        lat = row['cluster lat']
        long = row['cluster long']

        # iterate over zoom level list
        for index, zoom_level in enumerate(
            zoom_level_lats
        ):

            # get the path to the respective zoom level folder
            path_to_zoom_level_folder = (
                path_to_cluster_image_folder 
                + zoom_level 
                + '/'
            )

            # get the lat and long zoom parameters
            lat_zoom_param = zoom_level_lats[
                zoom_level
            ]
            long_zoom_param = long_var_deg_list[
                index
            ]

            # calculate the coordinates that you want to use for getting an excerpt of the image
            lat_min = lat - lat_zoom_param
            lat_max = lat + lat_zoom_param
            long_min = long - long_zoom_param
            long_max = long + long_zoom_param

            # create the bounding box parameter string
            bbox = (
                'BBOX=' 
                + str(lat_min) 
                + ',' 
                + str(long_min) 
                + ',' 
                + str(lat_max) 
                + ',' 
                + str(long_max)
            )

            # create the urls for requests
            url_ortho = (
                api_base_ortho 
                + fixed_params_ortho 
                + bbox
            )
            url_dem_relief = (
                api_base_dem 
                + fixed_params_dem_relief 
                + bbox
            )
            url_dem_aspect = (
                api_base_dem 
                + fixed_params_dem_aspect 
                + bbox
            )
            url_dem_slope = (
                api_base_dem 
                + fixed_params_dem_slope 
                + bbox
            )

            # create the file names for saving the requested images
            file_name_ortho = (
                'ortho_' 
                + zoom_level 
                + '_' 
                + str(cluster_id) 
                + '.png'
            )
            file_name_relief = (
                'relief_' 
                + zoom_level 
                + '_' 
                + str(cluster_id) 
                + '.png'
            )
            file_name_aspect = (
                'aspect_' 
                + zoom_level 
                + '_' 
                + str(cluster_id) 
                + '.png'
            )
            file_name_slope = (
                'slope_' 
                + zoom_level 
                + '_' 
                + str(cluster_id) 
                + '.png'
            )


            ### 
            # request the images and save these in the results folder ###
            ###

            # paths to respective image types
            path_to_ortho_folder = (
                path_to_zoom_level_folder 
                + 'ortho/'
            )
            path_to_aspect_folder = (
                path_to_zoom_level_folder 
                + 'aspect/'
            )
            path_to_slope_folder = (
                path_to_zoom_level_folder 
                + 'slope/'
            )
            path_to_relief_folder = (
                path_to_zoom_level_folder 
                + 'relief/'
            )

            # list of already requested images
            list_of_saved_ortho = os.listdir(
                path_to_ortho_folder
            )
            list_of_saved_aspect = os.listdir(
                path_to_aspect_folder
            )
            list_of_saved_slope = os.listdir(
                path_to_slope_folder
            )
            list_of_saved_relief = os.listdir(
                path_to_relief_folder
            )

            # ortho
            if file_name_ortho not in list_of_saved_ortho:

                resp = requests.get(
                    url_ortho, 
                    stream=True
                ).content
                image = np.asarray(
                    bytearray(resp), 
                    dtype="uint8"
                )
                image = cv2.imdecode(
                    image, 
                    cv2.IMREAD_COLOR
                )
                path_to_output_file = (
                    path_to_ortho_folder 
                    + file_name_ortho
                )
                if save_results:
                    cv2.imwrite(
                        path_to_output_file, 
                        image
                    )

            # relief
            if file_name_relief not in list_of_saved_relief:
                resp = requests.get(url_dem_relief, stream=True).content
                image = np.asarray(bytearray(resp), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                path_to_output_file = path_to_relief_folder + file_name_relief
                if save_results:
                    cv2.imwrite(path_to_output_file, image)

            # aspect
            if file_name_aspect not in list_of_saved_aspect:
                resp = requests.get(url_dem_aspect, stream=True).content
                image = np.asarray(bytearray(resp), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                path_to_output_file = path_to_aspect_folder + file_name_aspect
                if save_results:
                    cv2.imwrite(path_to_output_file, image)

            # slope
            if file_name_slope not in list_of_saved_slope:
                resp = requests.get(url_dem_slope, stream=True).content
                image = np.asarray(bytearray(resp), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                path_to_output_file = path_to_slope_folder + file_name_slope
                if save_results:
                    cv2.imwrite(path_to_output_file, image)


    ###
    # Visualize images ###
    ###

    # create subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(20,20))

    resp = requests.get(url_ortho, stream=True).content
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Ortho imagery', fontsize=20)

    resp = requests.get(url_dem_relief, stream=True).content
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    axs[0, 1].imshow(image)
    axs[0, 1].set_title('relief imagery', fontsize=20)

    resp = requests.get(url_dem_aspect, stream=True).content
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    axs[1, 0].imshow(image)
    axs[1, 0].set_title('aspect imagery', fontsize=20)

    resp = requests.get(url_dem_slope, stream=True).content
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    axs[1, 1].imshow(image)
    _ = axs[1, 1].set_title('slope imagery', fontsize=20)    
    
    
def get_building_aerial_imagery(
    profile_set
):

    # set the path to the raw building imagery
    path_to_raw_building_image_folder = (
        '../data/private/' + profile_set + '/building imagery/raw/'
    )


    ### Visualize exemplar images ###

    # create subplot figure
    fig, axs = plt.subplots(
        2, 
        2, 
        figsize=(20,20), 
        sharex=True, 
        sharey=True
    )

    # set figure title
    fig.suptitle(
        'Exemplar building-scale aerial imagery', 
        fontsize=20
    )

    # create path to geotiff files
    path_to_image1 = (
        path_to_raw_building_image_folder 
        + 'building 1.tif'
    )
    path_to_image2 = (
        path_to_raw_building_image_folder 
        + 'building 2.tif'
    )
    path_to_image3 = (
        path_to_raw_building_image_folder 
        + 'building 3.tif'
    )
    path_to_image4 = (
        path_to_raw_building_image_folder 
        + 'building 4.tif'
    )

    # import geotiff files
    ds1 = rasterio.open(path_to_image1)
    ds2 = rasterio.open(path_to_image2)
    ds3 = rasterio.open(path_to_image3)
    ds4 = rasterio.open(path_to_image4)

    # get rasters as numeric arrays
    rb_list1, rb_list2, rb_list3, rb_list4 = [],[],[],[]
    for i in range(3):
        
        rb_list1.append(ds1.read(i+1))
        rb_list2.append(ds2.read(i+1))
        rb_list3.append(ds3.read(i+1))
        rb_list4.append(ds4.read(i+1))
        
        
    # expand dimension for adding channels
    img_array1 = np.expand_dims(rb_list1[0], 2)
    img_array2 = np.expand_dims(rb_list2[0], 2)
    img_array3 = np.expand_dims(rb_list3[0], 2)
    img_array4 = np.expand_dims(rb_list4[0], 2)

    # add further channels to get RGB images
    for i in range(2):
        
        img_array1 = np.concatenate(
            [
                img_array1, 
                np.expand_dims(
                    rb_list1[i+1], 
                    2
                )
            ], 
            2
        )
        img_array2 = np.concatenate(
            [
                img_array2, 
                np.expand_dims(
                    rb_list2[i+1], 
                    2
                )
            ], 
            2
        )
        img_array3 = np.concatenate(
            [
                img_array3, 
                np.expand_dims(
                    rb_list3[i+1], 
                    2
                )
            ], 
            2
        )
        img_array4 = np.concatenate(
            [
                img_array4, 
                np.expand_dims(
                    rb_list4[i+1], 
                    2
                )
            ], 
            2
        )

    # set axis limits to plot all images to same scale
    axs[0, 0].set_xlim([0, 170])
    axs[0, 0].set_ylim([0, 170])

    # set subplot titles
    axs[0, 0].set_title(
        'building with ID 1', 
        fontsize=18
    )
    axs[0, 1].set_title(
        'building with ID 2', 
        fontsize=18
    )
    axs[1, 0].set_title(
        'building with ID 3', 
        fontsize=18
    )
    axs[1, 1].set_title(
        'building with ID 4', 
        fontsize=18
    )

    # plot images
    axs[0, 0].imshow(img_array1)
    axs[0, 1].imshow(img_array2)
    axs[1, 0].imshow(img_array3)
    _ = axs[1, 1].imshow(img_array4)
    
    
def pad_building_aerial_imagery(
    profile_set,
    save_results=False
):

    ###
    # Set parameters ###
    ###

    # set the path to the raw building imagery
    path_to_raw_building_image_folder = (
        '../data/private/' + profile_set + '/building imagery/raw/'
    )

    # set the path to the folder where padded building imagery should be saved
    path_to_padded_building_image_folder = (
        '../data/private/' + profile_set + '/building imagery/padded/'
    )


    ###
    # Calculate the desired padding size first ###
    ###

    # read the image file names
    building_image_list = os.listdir(
        path_to_raw_building_image_folder
    )

    # create an array to save the sizes of images
    img_size_array = np.zeros(
        (
            len(building_image_list), 
            2
        )
    )

    # iterate over each image
    for index, building_image_file in enumerate(
        building_image_list
    ):
        
        # create path to currently iterated file
        path_to_file = (
            path_to_raw_building_image_folder 
            + building_image_file
        )
        
        # import image
        image = rasterio.open(path_to_file)
        
        # get image sizes
        x_size = image.width
        y_size = image.height
        
        # save image sizes to array
        img_size_array[index] = [
            x_size, 
            y_size
        ]
        
    # calculate the maximum sizes which are the desired padding targets
    max_y, max_x = (
        img_size_array.max(axis=0).astype(int)
    )


    ###
    # Pad all images to desired target size ###
    ###


    # iterate over each image
    for index, building_image_file in enumerate(
        building_image_list
    ):

        # create path to currently iterated file
        path_to_file = (
            path_to_raw_building_image_folder 
            + building_image_file
        )

        # create path to building image output file
        path_to_output_file = (
            path_to_padded_building_image_folder 
            + building_image_file[:-4] 
            + '.png'
        )
        
        # import image
        image = rasterio.open(path_to_file)

        # get first band
        numeric_image = image.read(1)
        
        # expand dims
        numeric_image = np.expand_dims(
            numeric_image, 
            2
        )
        
        # add the second band
        numeric_image = np.concatenate(
            (
                numeric_image, 
                np.expand_dims(
                    image.read(2), 
                    2
                )
            ), 
            axis=2
        )
        
        # add the third band
        numeric_image = np.concatenate(
            (
                numeric_image, 
                np.expand_dims(
                    image.read(3), 
                    2
                )
            )
            , 
            axis=2
        )
        
        # get the borders right
        top = math.floor(
            (
                max_x - numeric_image.shape[0]
            )/2
        )
        bottom = (
            max_x - 
            top - 
            numeric_image.shape[0]
        )
        left = math.floor(
            (
                max_y - numeric_image.shape[1]
            )/2
        )
        right = ( 
            max_y 
            - left 
            - numeric_image.shape[1]
        )

        # pad the image
        numeric_image = cv2.copyMakeBorder(
            numeric_image, 
            top, 
            bottom, 
            left, 
            right, 
            cv2.BORDER_CONSTANT
        )

        # save the image
        if save_results:
            cv2.imwrite(
                path_to_output_file, 
                numeric_image
            )
        
        
    ### Plot exemplar padded images ###
    # read the image file names
    building_image_file_list = os.listdir(
        path_to_padded_building_image_folder
    )

    # create empty list for saving image imports
    building_imagery_list = []
    building_id_list = []

    # iterate over the first four images
    for i in range(4):
        
        # get currently iterated file name
        file_name = building_image_file_list[i]
        
        # get building id of currently iterated image
        building_id = int(file_name[:-4][9:])
        
        # create full path to .png image
        path_to_file = (
            path_to_padded_building_image_folder 
            + file_name
        )

        # import image
        image = Image.open(path_to_file)
        
        # add image to list of images
        building_imagery_list.append(image)
        building_id_list.append(building_id)
        
    # create subplot figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
        
    ax[0, 0].imshow(building_imagery_list[0])
    ax[0, 1].imshow(building_imagery_list[1])
    ax[1, 0].imshow(building_imagery_list[2])
    ax[1, 1].imshow(building_imagery_list[3])

    ax[0, 0].set_title(
        'building with ID ' + str(
            building_id_list[0]
        )
        , 
        fontsize=18
    )
    ax[0, 1].set_title(
        'building with ID ' + str(
            building_id_list[1]
        ), 
        fontsize=18
    )
    ax[1, 0].set_title(
        'building with ID ' + str(
            building_id_list[2]
        ), 
        fontsize=18
    )
    _ = ax[1, 1].set_title(
        'building with ID ' + str(
            building_id_list[3]
        ), 
        fontsize=18
    )
    
    

def create_image_pixel_data(
    profile_set,
    save_results=False
):
    
    ### Set some parameters ###

    # decide how many histogram bins to create
    histo_bins = 100

    # set the range of the histogram bins
    histo_range = (0, 1)

    # provide the paths to input and output folders
    path_to_building_image_folder = (
        '../data/private/' + profile_set + '/building imagery/padded/'
    )
    path_to_pixel_histogram_folder = (
        '../data/public/' + profile_set + '/building imagery/histogram/'
    )
    path_to_pixel_average_folder = (
        '../data/public/' + profile_set + '/building imagery/average/'
    )

    # read the image file names
    building_image_list = os.listdir(
        path_to_building_image_folder
    )

    # create dataframes for saving values
    df_average = pd.DataFrame()
    df_average_grey = pd.DataFrame()
    df_histogram = pd.DataFrame()
    df_histogram_grey = pd.DataFrame()

    # iterate over all building image files
    for file_name in building_image_list:
        
        # get building id of currently iterated image
        building_id = int(file_name[:-4][9:])
        
        # create full path to file
        path_to_file = (
            path_to_building_image_folder 
            + file_name
        )
        
        # import image
        image = Image.open(path_to_file)
        image_grey = image.convert("L")

        # transform the image to a numeric array
        image = np.asarray(image)
        image_grey = np.asarray(image_grey)

        # add channel to last axis
        image_grey = np.expand_dims(
            image_grey, 
            axis=image_grey.ndim
        )

        # normalize to 0 and 1 values. important to keep this for histogram bin range of 0-1 to work
        image = image / image.max()
        image_grey = image_grey / image_grey.max()

        # transform float64 to float32
        image = np.float32(image)
        image_grey = np.float32(image_grey)
        
        # get number of channels
        n_channels = image.shape[-1]
        n_channels_grey = image_grey.shape[-1]
        
        ### calculate pixel averages ###

        # create average array to save new values
        average_array = np.zeros(
            (1, n_channels)
        )
        average_array_grey = np.zeros(
            (1, n_channels_grey)
        )

        # iterate over all channels
        for i in range(n_channels):

            # calculate average of currently iterated channel
            average_array[:, i] = np.average(
                image[:, :, i], 
                axis=(0, 1)
            )
            
            if i == 0:
                
                average_array_grey[:, i] = np.average(
                    image_grey[:, :, i], 
                    axis=(0, 1)
                )
            
            
        ### calculate pixel histograms ###
        
        # create value array to save new values
        histogram_array = np.zeros(
            (histo_bins, n_channels)
        )
        histogram_array_grey = np.zeros(
            (histo_bins, n_channels_grey)
        )

        # iterate over all channels
        for i in range(n_channels):

            # calculate histogram of currently iterated channel
            histogram_array[:, i] = np.histogram(
                image[:, :, i],
                range=histo_range,
                bins=histo_bins,
            )[0]
            
            if i == 0:
                
                histogram_array_grey[:, i] = np.histogram(
                    image_grey[:, :, i],
                    range=histo_range,
                    bins=histo_bins,
                )[0]
        
        # flatten arrays
        average_array = average_array.flatten("F")
        average_array_grey = average_array_grey.flatten("F")
        histogram_array = histogram_array.flatten("F")
        histogram_array_grey = histogram_array_grey.flatten("F")
        
        # add pixel values to dataframes
        df_average[building_id] = average_array
        df_average_grey[building_id] = average_array_grey
        df_histogram[building_id] = histogram_array
        df_histogram_grey[building_id] = histogram_array_grey


    ### save results to csv files ###
    if save_results:
        path_to_file = (
            path_to_pixel_average_folder 
            + 'rgb/pixel_values.csv'
        )
        df_average.to_csv(
            path_to_file, 
            index=False, 
            header=True
        )

        path_to_file = (
            path_to_pixel_average_folder 
            + 'greyscale/pixel_values.csv'
        )
        df_average_grey.to_csv(
            path_to_file, 
            index=False, 
            header=True
        )

        path_to_file = (
            path_to_pixel_histogram_folder 
            + 'rgb/pixel_values.csv'
        )
        df_histogram.to_csv(
            path_to_file, 
            index=False, 
            header=True
        )

        path_to_file = (
            path_to_pixel_histogram_folder 
            + 'greyscale/pixel_values.csv'
        )
        df_histogram_grey.to_csv(
            path_to_file, 
            index=False, 
            header=True
        )
    
    
def generate_additional_building_year_profiles(
    profile_set,
    save_results=False
):


    ### Set some parameters ###

    path_to_building_images = '../data/private/' + profile_set + '/building imagery/raw/'
    path_to_original_profiles_folder = '../data/private/' + profile_set + '/building-year profiles/original/'
    path_to_minmax_profiles_folder = '../data/private/' + profile_set + '/building-year profiles/minmax/'
    path_to_featurescaled_profiles_folder = '../data/public/' + profile_set + '/building-year profiles/feature_scaled/'
    path_to_randomscaled_profiles_folder = '../data/public/' + profile_set + '/building-year profiles/random_scaled/'

    # get min-max scaler from sklearn preprocessing package
    min_max_scaler = preprocessing.MinMaxScaler()

    ### Create the scaled profiles ###

    # get the list of all original building-year profiles
    building_year_profile_list = os.listdir(path_to_original_profiles_folder)

    # iterate over the list
    for file in building_year_profile_list:
        
        # get the full path to currently iterated file
        path = path_to_original_profiles_folder + file
        
        # import as pandas dataframe
        df_original = pd.read_csv(path)
        df_minmax = df_original.copy()
        
        ### create minmax profiles ###
        
        # min-max scale
        df_minmax.iloc[2:, 1:] = (
            min_max_scaler.fit_transform(
                df_minmax.iloc[2:, 1:]
            )
        )
        
        # save the results
        if save_results:
            path_to_results = (
                path_to_minmax_profiles_folder 
                + file
            )
            df_minmax.to_csv(
                path_to_results, 
                index=False, 
                header=True
            )

        # create copies of df with min-max scaled profiles
        df_random = df_minmax.copy()
        df_public = df_minmax.copy()
        
        # get building id list
        building_id_list = df_minmax.columns[1:]
        
        # get number of building year profiles
        n_profiles = len(building_id_list)
        
        
        # iterate over all profiles/columns in df
        for i in range(n_profiles):
            
            ### create random scaling factor ###
            
            # choose a random scale factor between 1.5 and 90
            random_scale_factor = (
                random.randrange(10, 90)
            )
            
            # with 50% chance, divide this scaling factor by 100
            if random.choice([True, False]):
                
                random_scale_factor = (
                    random_scale_factor / 100
                )
                
            ### create proportional scaling factor ###
            
            # get currently iterated building id
            building_id = building_id_list[i]
            
            # get image file name
            image_name = (
                "building " 
                + str(building_id)
                + ".tif"
            )
            
            # create path to image file 
            path_to_image = (
                path_to_building_images 
                + image_name
            )
            
            # import image
            image = rasterio.open(path_to_image)

            # get first band
            numeric_image = image.read(1)
            
            # create a scaling factor proportional to 10 times number of pixels divided by maximum number of pixels
            prop_scale_factor = (
                100 
                * numeric_image.shape[0] 
                * numeric_image.shape[1] 
                / (408 * 456)
            )
            
            ### scale profiles ###
            
            # scale currently iterated profile by random scaling factor
            df_random.iloc[2:, i+1] = (
                df_random.iloc[2:, i+1] 
                * random_scale_factor
            )
            
            # scale currently iterated profile by 
            df_public.iloc[2:, i+1] = (
                df_public.iloc[2:, i+1] 
                * prop_scale_factor
            )
            
            
        if save_results:
            # save random results
            path_to_results = (
                path_to_randomscaled_profiles_folder 
                + file
            )
            df_random.to_csv(
                path_to_results, 
                index=False, 
                header=True
            )
        
            # save proportional results
            path_to_results = (
                path_to_featurescaled_profiles_folder 
                + file
            )
            df_public.to_csv(
                path_to_results, 
                index=False, 
                header=True
            )
        
    # Plot exemplar results 
    n_subplots = min(len(df_minmax.columns), 10)
    _ = df_minmax.iloc[2:, 1:n_subplots+1].plot(
        subplots=True, 
        layout=(math.ceil(n_subplots/2), 2), 
        figsize=(16, n_subplots)
    )
