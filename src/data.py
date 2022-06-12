import datetime
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage.transform import rescale
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class RawData:

    """ Keeps imported data, their paths and visualization parameters together.
    """

    def __init__(self, HYPER):
    
        """ Initializes paths and miscellaneous other values """
    
        # provide the path to where data is stored
        path_to_data = '../data/'
        
        # provide the path to where images are stored
        self.path_to_images = '../images/'
        if not os.path.exists(self.path_to_images):
            os.mkdir(self.path_to_images)
        
        # provide the saving path to where computational graph images are stored
        self.path_to_computational_graphs = self.path_to_images + 'computational graphs/'
        if not os.path.exists(self.path_to_computational_graphs):
            os.mkdir(self.path_to_computational_graphs)
        
        # determines how many exemplar subplots to show for load profiles
        self.n_subplots = 10

        # set the range of the histogram bins and the total number of bins.
        self.histo_range = (0, 1)
        
        # set the number of channels
        if HYPER.GREY_SCALE:
            self.n_channels = 1
        else:
            self.n_channels = 3

        # set the path to electric load profile data
        if HYPER.LABELS == 'feature_scaled' or HYPER.LABELS == 'random_scaled':
            self.path_to_building_year_profile_folder = (
                path_to_data  
                + 'public/'
                + HYPER.PROFILE_SET
                + '/building-year profiles/'
                + HYPER.LABELS 
                + '/'
            )
        else:
            self.path_to_building_year_profile_folder = (
                path_to_data  
                + 'private/'
                + HYPER.PROFILE_SET
                + '/building-year profiles/'
                + HYPER.LABELS 
                + '/'
            )
        
        # set the path to meteo data
        self.path_to_meteo_data_folder = (
            path_to_data 
            + 'public/'
            + HYPER.PROFILE_SET
            + '/meteo data/'
        )
        
        # set the path to aerial imagery data
        if HYPER.PRIVATE_DATA_ACCESS:
            self.path_to_aerial_imagery_folder = (
                path_to_data
                + 'private/'
                + HYPER.PROFILE_SET
                + '/building imagery/' 
                + 'padded/'
            )
        else:
        
            if HYPER.SPATIAL_FEATURES == 'histogram':
                self.path_to_aerial_imagery_folder = (
                    path_to_data 
                    + 'public/'
                    + HYPER.PROFILE_SET
                    + '/building imagery/'
                    + 'histogram/'
                )
                
            elif HYPER.SPATIAL_FEATURES == 'average':
                self.path_to_aerial_imagery_folder = (
                    path_to_data 
                    + 'public/'
                    + HYPER.PROFILE_SET
                    + '/building imagery/'
                    + 'average/'
                )
            
            if HYPER.GREY_SCALE:
                self.path_to_aerial_imagery_folder = (
                    self.path_to_aerial_imagery_folder 
                    + 'greyscale/'
                )
            else:
                self.path_to_aerial_imagery_folder = (
                    self.path_to_aerial_imagery_folder 
                    + 'rgb/'
                )
            
        
        ### Set the path to the folder for saving temporary trained encoders ###     
        self.path_to_tmp_encoder_weights = '../tmp encoder weights/'
        if not os.path.exists(self.path_to_tmp_encoder_weights):
            os.mkdir(self.path_to_tmp_encoder_weights)
            
        if HYPER.SAVE_RESULTS:
        
            # create a results folder if not existent
            path_to_results = '../results/'
            if not os.path.exists(path_to_results):
                os.mkdir(path_to_results)
            path_to_results += HYPER.PROFILE_SET + '/'
            if not os.path.exists(path_to_results):
                os.mkdir(path_to_results)
            
            # create the experiment name string for saving models and results
            if HYPER.RED_CAND_DATA_ACT_LRN:
                self.experiment_name = 'delta1'
            else:
                self.experiment_name = 'delta0'
            if HYPER.UPD_VAL_DATA_ACT_LRN:
                self.experiment_name += '_valup1'
            else:
                self.experiment_name += '_valup0'

            # create path for saving numerical results
            self.path_to_AL_results = path_to_results + 'values/'
            if not os.path.exists(self.path_to_AL_results):
                os.mkdir(self.path_to_AL_results)
            self.path_to_AL_results += self.experiment_name + '/'
            if not os.path.exists(self.path_to_AL_results):
                os.mkdir(self.path_to_AL_results)
                
            if HYPER.TEST_EXPERIMENT_CHOICE == 'main_experiments':
            
                # create path for saving models            
                self.path_to_AL_models = path_to_results +'models/'
                if not os.path.exists(self.path_to_AL_models):
                    os.mkdir(self.path_to_AL_models)
                self.path_to_AL_models += self.experiment_name + '/'
                if not os.path.exists(self.path_to_AL_models):
                    os.mkdir(self.path_to_AL_models)
                
                # create path for saving sample data points
                self.path_to_AL_test_samples = path_to_results + 'samples/'
                if not os.path.exists(self.path_to_AL_test_samples):
                    os.mkdir(self.path_to_AL_test_samples)
                self.path_to_AL_test_samples += self.experiment_name + '/'
                if not os.path.exists(self.path_to_AL_test_samples):
                    os.mkdir(self.path_to_AL_test_samples)
        
        
      

    def show_attributes(self):

        """ Prints out the attribute names of this class when called.
        """

        for attr, value in self.__dict__.items():
            print(attr)


class Dataset:

    """ Keeps a dataset together that contains multiple elements of X_t, X_s, 
    X_s1, X_st and Y.
    """

    def __init__(self, X_t_ord_1D, X_t, X_s, X_s1, X_st, Y):

        """ Initializes a complete set of attributes for a new Dataset object. 
        Note that missing values should conventionally be passed with a zero.
        """

        self.X_t_ord_1D = X_t_ord_1D
        self.X_t = X_t
        self.X_s = X_s
        self.X_s1 = X_s1
        self.X_st = X_st
        self.Y = Y

        self.n_datapoints = len(X_t)

    def randomize(self):

        """ Randomizes all data entries.
        """

        # create random array
        random_array = np.arange(len(self.X_t))

        # shuffle random array
        np.random.shuffle(random_array)

        if type(self.X_t_ord_1D) != int and type(self.X_t_ord_1D) != float:
            self.X_t_ord_1D = self.X_t_ord_1D[random_array]

        if type(self.X_t) != int and type(self.X_t) != float:
            self.X_t = self.X_t[random_array]
            
        if type(self.X_s) != int and type(self.X_s) != float:
            self.X_s = self.X_s[random_array]
            
        if type(self.X_s1) != int and type(self.X_s1) != float:
            self.X_s1 = self.X_s1[random_array]
            
        if type(self.X_st) != int and type(self.X_st) != float:
            self.X_st = self.X_st[random_array]
            
        if type(self.Y) != int and type(self.Y) != float:
            self.Y = self.Y[random_array]
            
        if hasattr(self, "Y_copy"):
            if type(self.Y_copy) != int and type(self.Y_copy) != float:
                self.Y_copy = self.Y_copy[random_array]

    def show_attributes(self):

        """ Prints out the attribute names of this class when called.
        """

        for attr, value in self.__dict__.items():
            print(attr)


def import_consumption_profiles(HYPER, raw_data, silent=False, plot=True):

    """ Imports consumption profiles and appends the following lists to the 
    raw_data object: building_year_profiles_list, building_id_list, 
    cluster_id_list, year_id_list, building_id_set, cluster_id_set, year_id_set, 
    cluster_year_set.
    """

    if not silent:
        # tell us what we are doing
        print('Importing consumption profiles')

        # create a progress bar
        progbar = tf.keras.utils.Progbar(len(HYPER.PROFILE_YEARS))

    # save dataframes here instead of under distinct names
    building_year_profiles_list = []
    memory_demand_GB = 0

    # iterate over the list of years for which we want to import load profiles
    for index_year, year in enumerate(HYPER.PROFILE_YEARS):

        # get the path to currently iterated building-year profiles file
        path_to_building_year_profile_files = (
            raw_data.path_to_building_year_profile_folder
            + str(year)
            + ' building-year profiles.csv'
        )

        # load currently iterated file
        df = pd.read_csv(path_to_building_year_profile_files)

        # get the building IDs of profiles
        building_ids = df.columns.values[1:]

        # get the cluster IDs of profiles and drop the row
        cluster_ids = df.iloc[0, 1:].values.astype(int)

        # get the years of profiles and replace them with the year ID used here
        years = df.iloc[1, 1:].values.astype(int)
        year_ids = years
        year_ids[:] = index_year

        # drop the cluder id and year rows
        df = df.drop([0, 1])

        # rename the 'building ID' column name to 'local_time' so as to match 
        # the meteo files' column name for search later
        df = df.rename(columns={'building ID': 'local_time'})

        # get the time stamp of the imported meters
        time_stamp_profiles = df.pop('local_time')

        # set the new time stamp as index
        df = df.set_index(time_stamp_profiles)

        # create a random array
        randomize = np.arange(len(building_ids))
        np.random.shuffle(randomize)

        # shuffle ID orders with same random array
        building_ids = building_ids[randomize]
        cluster_ids = cluster_ids[randomize]
        year_ids = year_ids[randomize]
        
        # shorten the considered ID lists according to your chosen number of  
        # considerable profiles per year
        n_profiles = math.ceil(HYPER.PROFILES_PER_YEAR * len(building_ids))
        building_ids = building_ids[: n_profiles]
        cluster_ids = cluster_ids[: n_profiles]
        year_ids = year_ids[: n_profiles]

        # shorten dataframe accordingly
        df = df[building_ids]

        # check if first iteration
        if year == HYPER.PROFILE_YEARS[0]:

            # if yes, set the id lists equal to currently iterated lists
            building_id_list = building_ids
            cluster_id_list = cluster_ids
            year_id_list = year_ids

        else:

            # if not, concatenate previous lists with currently iterated lists
            building_id_list = np.concatenate((building_id_list, building_ids))
            cluster_id_list = np.concatenate((cluster_id_list, cluster_ids))
            year_id_list = np.concatenate((year_id_list, year_ids))

        # append dataframe
        building_year_profiles_list.append(df)

        # accumulate the memory demand of building-year profiles we imported
        memory_demand_GB = memory_demand_GB + df.memory_usage().sum() * 1e-9

        if not silent:
            # increment the progress bar
            progbar.add(1)

    # get the set of building IDs, i.e. drop the duplicate entries
    building_id_set = set(building_id_list)

    # get the set of building IDs, i.e. drop the duplicate entries
    cluster_id_set = set(cluster_id_list)

    # get the set of year IDs. Note: this should be equal to PROFILE_YEARS
    year_id_set = set(year_id_list)

    # get set of cluster-year ID combinations
    cluster_year_set = set(list(zip(cluster_id_list, year_id_list)))

    raw_data.building_year_profiles_list = building_year_profiles_list
    raw_data.building_id_list = building_id_list
    raw_data.cluster_id_list = cluster_id_list
    raw_data.year_id_list = year_id_list
    raw_data.building_id_set = building_id_set
    raw_data.cluster_id_set = cluster_id_set
    raw_data.year_id_set = year_id_set
    raw_data.cluster_year_set = cluster_year_set

    # Tell us how much RAM we are occupying with the just imported profiles
    print(
        'The',
        len(building_id_list),
        'imported electric load profiles demand a total amount of',
        memory_demand_GB,
        'GB of RAM',
    )

    if plot:

        # set the number of subplots to the minimum of the desired value and the  
        # actually available profiles for plotting
        n_subplots = min(raw_data.n_subplots, len(df.columns))

        # Visualize some profiles
        _ = df.iloc[:, :n_subplots].plot(
            title='Exemplar electric load profiles (labels/ground truth data)',
            subplots=True,
            layout=(math.ceil(n_subplots / 2), 2),
            figsize=(16, n_subplots),
        )

    return raw_data


def import_building_images(HYPER, raw_data, silent=False, plot=True):

    """ Imports building-scale aerial imagery and appends the following to the 
    raw_data object: building_imagery_data_list, building_imagery_id_list.
    """

    if not silent:

        # tell us what we do
        print('Importing building-scale aerial imagery:')

        # create a progress bar
        progbar = tf.keras.utils.Progbar(len(raw_data.building_id_set))

        # create a variabl to iteratively add the memory of imported files
        memory_demand_GB = 0

    # create a empty lists for aerial image data and building ids
    building_imagery_data_list = []
    building_imagery_id_list = []

    if HYPER.PRIVATE_DATA_ACCESS:
        
        # iterate over set of building ID
        for building_id in raw_data.building_id_set:

            # get the file name first
            file_name = 'building ' + building_id + '.png'

            # create the entire path to the currently iterated file
            path_to_file = raw_data.path_to_aerial_imagery_folder + file_name

            # import image
            image = Image.open(path_to_file)

            # convert to grey scale if this is chosen so and add channel
            if HYPER.GREY_SCALE == True:

                # convert to grey-scale
                image = image.convert('L')

                # transform the image to a numeric array
                image = np.asarray(image)

                # add channel to last axis
                image = np.expand_dims(image, axis=image.ndim)

            else:

                # transform the image to a numeric array
                image = np.asarray(image)

            # down-scale image if this is chosen, note that image array entries 
            # are rounded to integer again to save RAM. Before up-scaling, 
            # multiplication with a large factor ensured that rounding errors 
            # are small, and one can again down-sample with the same factor 
            # during training with no loss of RAM and information
            if HYPER.DOWN_SCALE_BUILDING_IMAGES != 0:

                # note that downscaled have float entries instead of integers
                image = rescale(
                    image,
                    1 / HYPER.DOWN_SCALE_BUILDING_IMAGES,
                    anti_aliasing=False,
                    multichannel=True,
                )

            # normalize to 0 and 1 values
            image = image / image.max()

            # transform float64 to float32
            image = np.float32(image)

            # add values to lists
            building_imagery_id_list.append(int(building_id))
            building_imagery_data_list.append(image)

            if not silent:

                # Accumulate the memory demand of each image
                memory_demand_GB = memory_demand_GB + image.nbytes * 1e-9

                # increment progress bar
                progbar.add(1)

        if plot:

            # create subplot figure and axes
            fig, ax = plt.subplots(2, 2, figsize=(16, 16))
        
            # set title
            fig.suptitle('Exemplar building images (spatial features)')

            # plot images
            for i_img in range(4):
            
                i_row = math.floor(i_img/2)
                i_col = int(i_img % 2)
                
                if HYPER.GREY_SCALE:
                    ax[i_row, i_col].imshow(building_imagery_data_list[i_img][:, :, 0])
                else:
                    ax[i_row, i_col].imshow(building_imagery_data_list[i_img])
          
                # add sub-titles
                _ = ax[i_row, i_col].set_title(building_imagery_id_list[i_img], fontsize=20)

    else:
    
        # create path to imagery data file
        path_to_file = (
            raw_data.path_to_aerial_imagery_folder 
            + 'pixel_values.csv'
        )

        # import building imagery data
        df = pd.read_csv(path_to_file)

        # iterate over set of building IDs
        for building_id in raw_data.building_id_set:
            
            # get the pixel features of currently iterated building image
            imagery_pixel_data = df[building_id].values
            
            ### reshape image pixel values a shape with channels last ###
 
            # get the number of features per image pixel array channel
            if HYPER.SPATIAL_FEATURES == 'average':
                n_features = 1
                
            elif HYPER.SPATIAL_FEATURES == 'histogram':
                n_features = HYPER.HISTO_BINS

            # reshape image with Fortran method. This is method used to flatten.
            imagery_pixel_data = np.reshape(
                imagery_pixel_data, 
                (n_features, raw_data.n_channels), 
                order='F'
            )

            # add values to lists
            building_imagery_data_list.append(imagery_pixel_data)
            building_imagery_id_list.append(int(building_id))

            if not silent:

                # Accumulate the memory demand of each image
                memory_demand_GB += imagery_pixel_data.nbytes * 1e-9

                # increment progress bar
                progbar.add(1)


    if not silent:

        # Tell us how much RAM we occupy with the just imported data files
        print(
            'The',
            len(building_imagery_data_list),
            'aerial images demand',
            memory_demand_GB,
            'GB RAM with float32 entries',
        )
 
    # add to raw_data instance
    raw_data.building_imagery_data_list = building_imagery_data_list
    raw_data.building_imagery_id_list = building_imagery_id_list

    return raw_data


def import_meteo_data(HYPER, raw_data, silent=False, plot=True):

    """ Imports cluster-scale meteorological data and appends the following 
    information to the raw_data object: meteo_data_list, 
    meteo_data_cluster_year_array.
    """

    if not silent:

        # tell us what we do
        print('Importing meteorological data')

        # create a variabl to iteratively add the memory demand of each file
        memory_demand_GB = 0

        # create a progress bar
        progbar = tf.keras.utils.Progbar(len(raw_data.cluster_year_set))

    # create list for saving meteo data
    meteo_data_list = []

    # create array for saving corresponding cluster and year IDs of meteo files
    # that are added to the list
    meteo_data_cluster_year_array = np.zeros(
        (
            len(raw_data.cluster_year_set), 
            2
        )
    )

    # use counter for meta data array
    counter = 0

    # iterate over each file in the list of all meteo files
    for cluster_id, year_id in raw_data.cluster_year_set:

        file_name = (
            'meteo_'
            + str(cluster_id)
            + '_'
            + str(int(HYPER.PROFILE_YEARS[year_id]))
            + '.csv'
        )

        # create the entire path to the currently iterated file
        path_to_file = raw_data.path_to_meteo_data_folder + file_name

        # load file
        df = pd.read_csv(path_to_file)

        # set one of the columns 'local_time' as index for later search purposes
        df = df.set_index('local_time')

        # shorten dataframe according to the meteo data types that you chose
        df = df[HYPER.METEO_TYPES]

        # append to list
        meteo_data_list.append(df)

        # append to list
        meteo_data_cluster_year_array[counter] = (cluster_id, year_id)

        # increment
        counter += 1

        if not silent:
            # Accumulate the memory demand of each file
            memory_demand_GB += df.memory_usage().sum() * 1e-9

            # increment progress bar
            progbar.add(1)

    raw_data.meteo_data_list = meteo_data_list
    raw_data.meteo_data_cluster_year_array = meteo_data_cluster_year_array

    if not silent:

        # Tell us how much RAM we occupy with the just imported data files
        print(
            'The',
            len(raw_data.cluster_year_set),
            'meteo data files demand',
            memory_demand_GB,
            'GB RAM',
        )

    if plot:

        # plot the time series data for each metering code
        _ = df.plot(
            title='Exemplar meteorological conditions (spatio-temporal features)',
            use_index=False,
            legend=True,
            figsize=(16, 16),
            fontsize=16,
            subplots=True,
            layout=(3, 3),
        )

    return raw_data


def create_feature_label_pairs(HYPER, raw_data, silent=False):

    """ Creates pairs of features and labels, and returns these bundled as a 
    Dataset object.
    """

    # determine start and end of iteration over each paired dataframe
    start = HYPER.HISTORY_WINDOW_METEO * 4
    end = (
        len(raw_data.building_year_profiles_list[0]) 
        - HYPER.PREDICTION_WINDOW
    )
    n_points = math.ceil(
        HYPER.POINTS_PER_PROFILE 
        * len(raw_data.building_year_profiles_list[0])
    )
    step = math.ceil((end - start) / n_points)
    points_per_profile = math.ceil((end - start) / step)

    # Calculate how many data points we chose to consider in total
    n_datapoints = len(raw_data.building_id_list) * points_per_profile

    # Create empty arrays in the right format for saving features and labels
    X_t = np.zeros((n_datapoints, 5))
    X_st = np.zeros(
        (
            n_datapoints, 
            HYPER.HISTORY_WINDOW_METEO, 
            len(HYPER.METEO_TYPES)
        )
    )
    X_s = np.zeros((n_datapoints, 2))
    Y = np.zeros((n_datapoints, HYPER.PREDICTION_WINDOW))

    # create a datapoint counter to increment and add to the data entries
    datapoint_counter = 0

    if not silent:

        # tell us what we do
        print('Creating feature label data pairs:')

        # create a progress bar
        progbar = tf.keras.utils.Progbar(n_datapoints)

    # iterate over the set of considered cluser-year ID combinations
    for cluster_id, year_id in raw_data.cluster_year_set:

        # generate the respective cluster id and building id subsets
        building_id_subset = raw_data.building_id_list[
            np.nonzero(
                (raw_data.year_id_list == year_id)
                & (raw_data.cluster_id_list == cluster_id)
            )
        ]

        # get the year in gregorian calendar here
        year = int(HYPER.PROFILE_YEARS[year_id])

        # get the index of the meteo data list entry that correspondings to 
        # the currently iterated cluster-year ID combination
        index_meteo_data_list = np.where(
            (raw_data.meteo_data_cluster_year_array[:, 0] == cluster_id)
            & (raw_data.meteo_data_cluster_year_array[:, 1] == year_id)
        )[0][0]

        # create a new dataframe that merges the meteo values and load profile
        # values by index col 'local_time'
        paired_df = raw_data.building_year_profiles_list[year_id][
            building_id_subset
        ].merge(raw_data.meteo_data_list[index_meteo_data_list], on="local_time")

        # iterate over the paired dataframe
        for i in range(start, end, step):

            # get timestamp features
            month = paired_df.index[i][5:7]
            day = paired_df.index[i][8:10]
            hour = paired_df.index[i][11:13]
            minute_15 = paired_df.index[i][14:16]

            # get the meteo features. Note that you need to jump in hourly
            # steps back in time, hence all times 4
            meteo = paired_df.iloc[
                (i - (HYPER.HISTORY_WINDOW_METEO * 4)) : i : 4,
                -(len(HYPER.METEO_TYPES)) :,
            ]

            # iterate over each building id
            for building_id in building_id_subset:

                # get the label
                label = (
                    paired_df[[building_id]]
                    .iloc[i : (i + HYPER.PREDICTION_WINDOW)]
                    .values[:, 0]
                )

                # Add the features and labels to respective data point entry
                X_t[datapoint_counter, :] = [minute_15, hour, day, month, year]
                X_s[datapoint_counter, :] = [building_id, cluster_id]
                X_st[datapoint_counter, :, :] = meteo
                Y[datapoint_counter, :] = label

                # increment datapoint counter
                datapoint_counter += 1

        if not silent:

            # increment progress bar
            progbar.add(points_per_profile * len(building_id_subset))


    ### Shorten X_t according to chosen TIMESTAMP_DATA ###

    # create empty list
    filter_list = []

    # check for all possible entries in correct order and add to filter list if 
    # not in chosen TIMESTAMP_DATA
    if '15min' not in HYPER.TIMESTAMP_DATA:
        filter_list.append(0)
        
    if 'hour' not in HYPER.TIMESTAMP_DATA:
        filter_list.append(1)
        
    if 'day' not in HYPER.TIMESTAMP_DATA:
        filter_list.append(2)
        
    if 'month' not in HYPER.TIMESTAMP_DATA:
        filter_list.append(3)
        
    if 'year' not in HYPER.TIMESTAMP_DATA:
        filter_list.append(4)

    # delete the columns according to created filter_list
    X_t = np.delete(X_t, filter_list, 1)

    # get the minimum value for labels
    raw_data.Y_min = Y.min()

    # get the maximum value for labels
    raw_data.Y_max = Y.max()

    # get the full range of possible values
    raw_data.Y_range = raw_data.Y_max - raw_data.Y_min

    # bundle data as dataset object and return
    dataset = Dataset(0, X_t, X_s, 0, X_st, Y)


    ### Process spatial features ###

    # check how we chose to consider spatial features
    if HYPER.SPATIAL_FEATURES != 'image':

        if HYPER.PRIVATE_DATA_ACCESS:

            ### Transform images to average/histogram arrays ###

            # check if averages are chosen to be used for spatial features
            if HYPER.SPATIAL_FEATURES == 'average':

                # iterate over all building-scale images
                for index, image in enumerate(
                    raw_data.building_imagery_data_list
                ):

                    # create value array to save new values
                    value_array = np.zeros((1, raw_data.n_channels))

                    # iterate over all channels
                    for i in range(raw_data.n_channels):

                        # calculate average of currently iterated channel
                        value_array[:, i] = np.average(
                            image[:, :, i], 
                            axis=(0, 1)
                        )

                    # assign value array of averages to currently iterated image
                    raw_data.building_imagery_data_list[index] = value_array

            # check if histograms are chosen to be used for spatial features
            elif HYPER.SPATIAL_FEATURES == 'histogram':

                # iterate over all building-scale images
                for index, image in enumerate(
                    raw_data.building_imagery_data_list
                ):

                    # create value array to save new values
                    value_array = np.zeros(
                        (
                            HYPER.HISTO_BINS, 
                            raw_data.n_channels
                        )
                    )

                    # iterate over all channels
                    for i in range(raw_data.n_channels):

                        # calculate histogram of currently iterated channel
                        value_array[:, i] = np.histogram(
                            image[:, :, i],
                            range=raw_data.histo_range,
                            bins=HYPER.HISTO_BINS,
                        )[0]

                    # assign value array of histograms to iterated image
                    raw_data.building_imagery_data_list[index] = value_array


        df_list = []

        # iterate over number of channels
        for i in range(raw_data.n_channels):

            # create dataframe with one column 'building id' for iterated channel
            df_list.append(pd.DataFrame(columns=['building id']))

        # iterate over all building scale images and their building IDs
        for index, image in enumerate(raw_data.building_imagery_data_list):

            building_id = raw_data.building_imagery_id_list[index]

            for channel, df in enumerate(df_list):
               
                df_list[channel] = df_list[channel].append(
                    pd.Series(image[:, channel]), ignore_index=True
                )

                df_list[channel].iloc[index, 0] = building_id

        # create empty X_s1
        dataset.X_s1 = np.zeros(
            (
                dataset.n_datapoints, 
                image.shape[0], 
                image.shape[1]
            )
        )

        # iterate over number of channels
        for i in range(raw_data.n_channels):

            # merge the columns of building ID in X_s and the new dataframe
            paired_df = pd.DataFrame(
                dataset.X_s, 
                columns=['building id', 'cluster id']
            ).merge(
                df_list[i], 
                on='building id', 
                how='left'
            )

            # pass the paired values to X_s1
            dataset.X_s1[:, :, i] = paired_df.iloc[:, 2:].values

    return dataset, raw_data


def encode_time_features(HYPER, dataset, silent=False):

    """ Takes X_t as input and returns it as ordinally encoded, One-Hot-Encoded, 
    or single dimensionally ordinally encoded array, according to hyper 
    parameter TIME_ENCODING.
    """

    if not silent:

        # tell us what we do
        print('Encoding temporal features')
        print('X_t before:', dataset.X_t[0])

    ###
    # Ordinally encode all available time stamp dimensions ###
    ###

    # get OrdinalEncoder from sklearn.preprocessing
    enc = OrdinalEncoder()

    # fit the encoder to X_t
    enc.fit(dataset.X_t)

    # encode X_t
    dataset.X_t = enc.transform(dataset.X_t).astype(int)

    # save the encoded feature categories for X_time
    timestamp_categories = enc.categories_

    # create empty matrix for saving number of categories of each feature column
    n_time_categories = np.zeros((len(enc.categories_))).astype(int)

    # iterate over each category array and save number of categories
    for index, category_array in enumerate(enc.categories_):

        # save number of respective category
        n_time_categories[index] = len(category_array)

    ###
    # Create one dimensional ordinal encoding in 1-min steps ###
    ###

    # create an empty array for adding up values
    dataset.X_t_ord_1D = np.zeros((dataset.n_datapoints,))
    X_t_copy = dataset.X_t

    # check for all possible entries
    if '15min' in HYPER.TIMESTAMP_DATA:
        dataset.X_t_ord_1D += X_t_copy[:, 0] * 15
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'hour' in HYPER.TIMESTAMP_DATA:
        dataset.X_t_ord_1D += X_t_copy[:, 0] * 60
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'day' in HYPER.TIMESTAMP_DATA:
        dataset.X_t_ord_1D += X_t_copy[:, 0] * 60 * 24
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'month' in HYPER.TIMESTAMP_DATA:
        dataset.X_t_ord_1D += X_t_copy[:, 0] * 60 * 24 * 31
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'year' in HYPER.TIMESTAMP_DATA:
        dataset.X_t_ord_1D += X_t_copy[:, 0] * 60 * 24 * 31 * 12
        X_t_copy = np.delete(X_t_copy, 0, 1)

    ###
    #  If chosen so, transform encoding here ###
    ###

    if HYPER.TIME_ENCODING == 'OHE':

        # get OHE encoder
        enc = OneHotEncoder()

        # fit encoder
        enc.fit(dataset.X_t)

        # encode temporal features
        dataset.X_t = enc.transform(dataset.X_t).toarray().astype(int)

    elif HYPER.TIME_ENCODING == 'ORD-1D':

        # copy the 1D ordinal array to X_t
        dataset.X_t = dataset.X_t_ord_1D

        # expand the last dimension for NN input fit
        dataset.X_t = np.expand_dims(dataset.X_t, axis=1)

    if not silent:

        print('X_t after: {} ({})'.format(dataset.X_t[0], HYPER.TIME_ENCODING))

    return dataset


def normalize_features(HYPER, raw_data, dataset, silent=False):

    """ Min-max normalizes all features if hyper parameter NORMALIZATION is set 
    True.
    """

    if HYPER.NORMALIZATION:

        if not silent:
        
            # tell us what we do
            print('Normalizing features')

        # get min-max scaler from the sklearn preprocessing package
        min_max_scaler = preprocessing.MinMaxScaler()

        # normalize X_t in the case that it is not OHE
        if HYPER.TIME_ENCODING != 'OHE':
            dataset.X_t = min_max_scaler.fit_transform(dataset.X_t)

        # normalize X_st
        for i in range(len(HYPER.METEO_TYPES)):
            dataset.X_st[:, :, i] = min_max_scaler.fit_transform(
                dataset.X_st[:, :, i]
            )

        # normalize X_s1
        if HYPER.SPATIAL_FEATURES != 'image':

            for channel in range(raw_data.n_channels):
                dataset.X_s1[:, :, channel] = min_max_scaler.fit_transform(
                    dataset.X_s1[:, :, channel]
                )

    return dataset


def split_train_val_test(HYPER, raw_data, dataset, silent=False):

    """ Splits passed dataset into trainin, validation and three different test 
    sets:
        1. spatial tests (building IDs that are not in training/validation data)
        2. temporal tests (time stamps not in training/validation data)
        3. spatio-temporal test (neither time stamp, nor building ID is in 
        training and validation data).
    """

    if not silent:
        # tell us what we are doing
        print('Splitting data into training, validation and testing sets.')

    ###
    # Reduce memory demand ###
    ###

    dataset.X_t = np.float32(dataset.X_t)
    dataset.X_st = np.float32(dataset.X_st)
    dataset.Y = np.float32(dataset.Y)
    dataset.X_s = dataset.X_s.astype(int)

    if HYPER.SPATIAL_FEATURES != 'image':
        dataset.X_s1 = np.float32(dataset.X_s1)

    ###
    # Sort arrays in ascending temporal order ###
    ###

    sort_array = np.argsort(dataset.X_t_ord_1D)
    dataset.X_t = dataset.X_t[sort_array]
    dataset.X_s = dataset.X_s[sort_array]
    dataset.X_st = dataset.X_st[sort_array]
    dataset.Y = dataset.Y[sort_array]

    if HYPER.SPATIAL_FEATURES != 'image':
        dataset.X_s1 = dataset.X_s1[sort_array]

    ###
    # Take away data from both ends of sorted arrays ###
    ###

    # get the number of datapoints to cut out for temporal prediction tests
    split_point = math.ceil(HYPER.TEST_SPLIT / 2 * dataset.n_datapoints)

    ### extract data from beginning of temporaly sorted dataset ###
    temporal_X_t_ord_1D = dataset.X_t_ord_1D[:split_point]
    dataset.X_t_ord_1D = dataset.X_t_ord_1D[split_point:]
    
    temporal_X_t = dataset.X_t[:split_point]
    dataset.X_t = dataset.X_t[split_point:]

    temporal_X_s = dataset.X_s[:split_point]
    dataset.X_s = dataset.X_s[split_point:]

    temporal_X_st = dataset.X_st[:split_point]
    dataset.X_st = dataset.X_st[split_point:]

    temporal_Y = dataset.Y[:split_point]
    dataset.Y = dataset.Y[split_point:]

    if HYPER.SPATIAL_FEATURES != 'image':
        temporal_X_s1 = dataset.X_s1[:split_point]
        dataset.X_s1 = dataset.X_s1[split_point:]

    ### extract data from end of temporaly sorted dataset ###
    temporal_X_t_ord_1D = np.concatenate(
        (
            temporal_X_t_ord_1D,
            dataset.X_t_ord_1D[-split_point:]
        )
    )
    dataset.X_t_ord_1D = dataset.X_t_ord_1D[:-split_point]
    
    temporal_X_t = np.concatenate(
        (
            temporal_X_t, 
            dataset.X_t[-split_point:]
        )
    )
    dataset.X_t = dataset.X_t[:-split_point]

    temporal_X_s = np.concatenate(
        (
            temporal_X_s, 
            dataset.X_s[-split_point:]
        )
    )
    dataset.X_s = dataset.X_s[:-split_point]

    temporal_X_st = np.concatenate(
        (
            temporal_X_st, 
            dataset.X_st[-split_point:]
        )
    )
    dataset.X_st = dataset.X_st[:-split_point]

    temporal_Y = np.concatenate(
        (
            temporal_Y, 
            dataset.Y[-split_point:]
        )
    )
    dataset.Y = dataset.Y[:-split_point]

    if HYPER.SPATIAL_FEATURES != 'image':
        temporal_X_s1 = np.concatenate(
            (
                temporal_X_s1, 
                dataset.X_s1[-split_point:]
            )
        )
        dataset.X_s1 = dataset.X_s1[:-split_point]


    ###
    # Sample building IDs for seperation as testing data ###
    ###
    
    # get number of buildings you want to randomly choose from
    n_test_buildings = math.ceil(
        HYPER.TEST_SPLIT * len(raw_data.building_id_set)
    )

    # randomly choose some buildings for testing
    test_building_samples = random.sample(
        raw_data.building_id_set, 
        k=n_test_buildings
    )

    # transform building ID strings to integers
    test_building_samples = [int(x) for x in test_building_samples]
    
    
    ###
    # Extract temporal and spatio-temporal test sets ###
    ###

    # writes True for all data points that are in test_building_samples
    boolean_filter_array = np.zeros((len(temporal_X_s),), dtype=bool)
    for building_id in test_building_samples:
        boolean_filter_array = boolean_filter_array | (
            temporal_X_s[:, 0] == building_id
        )
        
    # writes True for all data points that are not in test_building_samples
    inverted_boolean_filter_array = np.invert(boolean_filter_array)

    ### Spatio-temporal ###
    spatemp_X_t_ord_1D = temporal_X_t_ord_1D[boolean_filter_array]
    spatemp_X_t = temporal_X_t[boolean_filter_array]
    spatemp_X_s = temporal_X_s[boolean_filter_array]
    spatemp_X_st = temporal_X_st[boolean_filter_array]
    spatemp_Y = temporal_Y[boolean_filter_array]

    if HYPER.SPATIAL_FEATURES != 'image':
        spatemp_X_s1 = temporal_X_s1[boolean_filter_array]
    else:
        spatemp_X_s1 = 0

    spatemp_test_data = Dataset(
        spatemp_X_t_ord_1D,
        spatemp_X_t, 
        spatemp_X_s, 
        spatemp_X_s1, 
        spatemp_X_st, 
        spatemp_Y
    )
    (
        spatemp_X_t_ord_1D,
        spatemp_X_t, 
        spatemp_X_s, 
        spatemp_X_s1, 
        spatemp_X_st, 
        spatemp_Y
    ) = 0, 0, 0, 0, 0, 0

    ### Temporal ###
    temporal_X_t_ord_1D = temporal_X_t_ord_1D[inverted_boolean_filter_array]
    temporal_X_t = temporal_X_t[inverted_boolean_filter_array]
    temporal_X_s = temporal_X_s[inverted_boolean_filter_array]
    temporal_X_st = temporal_X_st[inverted_boolean_filter_array]
    temporal_Y = temporal_Y[inverted_boolean_filter_array]

    if HYPER.SPATIAL_FEATURES != 'image':
        temporal_X_s1 = temporal_X_s1[inverted_boolean_filter_array]
    else:
        temporal_X_s1 = 0

    temporal_test_data = Dataset(
        temporal_X_t_ord_1D,
        temporal_X_t, 
        temporal_X_s, 
        temporal_X_s1, 
        temporal_X_st, 
        temporal_Y
    )
    (
        temporal_X_t_ord_1D,
        temporal_X_t, 
        temporal_X_s, 
        temporal_X_s1, 
        temporal_X_st, 
        temporal_Y 
    ) = 0, 0, 0, 0, 0, 0


    ###
    # Extract spatial test set from train and validation ###
    ###

    # writes True for all remaining data points that are in test_building_samples
    boolean_filter_array = np.zeros((len(dataset.X_s),), dtype=bool)
    for building_id in test_building_samples:
        boolean_filter_array = (
            boolean_filter_array | (dataset.X_s[:, 0] == building_id)
        )

    # writes True for all remaining data points that are not in test_building_samples
    inverted_boolean_filter_array = np.invert(boolean_filter_array)

        
    ### Spatial ###
    spatial_X_t_ord_1D = dataset.X_t_ord_1D[boolean_filter_array]
    spatial_X_t = dataset.X_t[boolean_filter_array]
    spatial_X_s = dataset.X_s[boolean_filter_array]
    spatial_X_st = dataset.X_st[boolean_filter_array]
    spatial_Y = dataset.Y[boolean_filter_array]

    if HYPER.SPATIAL_FEATURES != 'image':
        spatial_X_s1 = dataset.X_s1[boolean_filter_array]
    else:
        spatial_X_s1 = 0

    spatial_test_data = Dataset(
        spatial_X_t_ord_1D,
        spatial_X_t, 
        spatial_X_s, 
        spatial_X_s1, 
        spatial_X_st, 
        spatial_Y
    )
    (
        spatial_X_t_ord_1D,
        spatial_X_t, 
        spatial_X_s, 
        spatial_X_s1, 
        spatial_X_st, 
        spatial_Y 
    ) = 0, 0, 0, 0, 0, 0


    ###
    # Split remaining into training and validation datasets using intervals ###
    ###
    
    ### Train-validation split ###
    train_val_X_t_ord_1D = dataset.X_t_ord_1D[inverted_boolean_filter_array]
    dataset.X_t_ord_1D = 0
    
    train_val_X_t = dataset.X_t[inverted_boolean_filter_array]
    dataset.X_t = 0
    
    train_val_X_s = dataset.X_s[inverted_boolean_filter_array]
    dataset.X_s = 0
    
    train_val_X_st = dataset.X_st[inverted_boolean_filter_array]
    dataset.X_st = 0
    
    train_val_Y = dataset.Y[inverted_boolean_filter_array]
    dataset.Y = 0

    if HYPER.SPATIAL_FEATURES != 'image':
        train_val_X_s1 = dataset.X_s1[inverted_boolean_filter_array]
        dataset.X_s1 = 0


    split_bins = math.ceil(len(train_val_X_t) * HYPER.SPLIT_INTERAVALS)
    random_array = np.arange(len(train_val_Y))
    random_array = np.array_split(random_array, split_bins)
    np.random.shuffle(random_array)
    random_array = np.concatenate(random_array).ravel()

    train_val_X_t_ord_1D = train_val_X_t_ord_1D[random_array]
    train_val_X_t = train_val_X_t[random_array]
    train_val_X_s = train_val_X_s[random_array]
    train_val_X_st = train_val_X_st[random_array]
    train_val_Y = train_val_Y[random_array]
    if HYPER.SPATIAL_FEATURES != 'image':
        train_val_X_s1 = train_val_X_s1[random_array]

    # get splitting point for training validation split
    split_point = math.ceil(HYPER.TRAIN_SPLIT * len(train_val_X_t))

    # split train and delete unused entries immediately
    X_t_ord_1D_train, X_t_ord_1D_val = np.split(train_val_X_t_ord_1D, [split_point])
    train_val_X_t_ord_1D = 0
    
    X_t_train, X_t_val = np.split(train_val_X_t, [split_point])
    train_val_X_t = 0

    X_s_train, X_s_val = np.split(train_val_X_s, [split_point])
    train_val_X_s = 0

    X_st_train, X_st_val = np.split(train_val_X_st, [split_point])
    train_val_X_st = 0

    Y_train, Y_val = np.split(train_val_Y, [split_point])
    train_val_Y = 0

    if HYPER.SPATIAL_FEATURES != 'image':
        X_s1_train, X_s1_val = np.split(train_val_X_s1, [split_point])
        train_val_X_s1 = 0

    else:
        X_s1_train, X_s1_val, X_s2_val = 0, 0, 0

    training_data = Dataset(
        X_t_ord_1D_train,
        X_t_train, 
        X_s_train, 
        X_s1_train, 
        X_st_train, 
        Y_train
    )
    (
        X_t_ord_1D_train,
        X_t_train, 
        X_s_train, 
        X_s1_train, 
        X_st_train, 
        Y_train
    ) = 0, 0, 0, 0, 0, 0

    validation_data = Dataset(
        X_t_ord_1D_val,
        X_t_val, 
        X_s_val, 
        X_s1_val, 
        X_st_val, 
        Y_val
    )
    (
        X_t_ord_1D_val,
        X_t_val, 
        X_s_val, 
        X_s1_val, 
        X_st_val, 
        Y_val
    ) = 0, 0, 0, 0, 0, 0

    training_data.randomize()
    validation_data.randomize()
    spatial_test_data.randomize()
    temporal_test_data.randomize()
    spatemp_test_data.randomize()

    if not silent:

        n_test_datapoints = (
            spatial_test_data.n_datapoints
            + temporal_test_data.n_datapoints
            + spatemp_test_data.n_datapoints
        )
        n_total_datapoints = (
            training_data.n_datapoints
            + validation_data.n_datapoints
            + n_test_datapoints
        )

        print(
            'With TRAIN_SPLIT =',
            HYPER.TRAIN_SPLIT,
            ' and TEST_SPLIT =',
            HYPER.TEST_SPLIT,
            'the data is split in the following ratio:',
        )
        print('---' * 38)

        print(
            'Training data:   {} ({:.0%})'.format(
                training_data.n_datapoints,
                training_data.n_datapoints / n_total_datapoints,
            )
        )
        print(
            'Validation data: {} ({:.0%})'.format(
                validation_data.n_datapoints,
                validation_data.n_datapoints / n_total_datapoints,
            )
        )
        print(
            'Testing data:    {} ({:.0%})'.format(
                n_test_datapoints, n_test_datapoints / n_total_datapoints
            )
        )
        print('---' * 38)

        print(
            'Spatial testing data:         {} ({:.0%})'.format(
                spatial_test_data.n_datapoints,
                spatial_test_data.n_datapoints / n_test_datapoints,
            )
        )
        print(
            'Temporal testing data:        {} ({:.0%})'.format(
                temporal_test_data.n_datapoints,
                temporal_test_data.n_datapoints / n_test_datapoints,
            )
        )
        print(
            'Spatio-temporal testing data: {} ({:.0%})'.format(
                spatemp_test_data.n_datapoints,
                spatemp_test_data.n_datapoints / n_test_datapoints,
            )
        )

    if HYPER.PRED_TYPE_ACT_LRN=='spatial':
        testing_data = spatial_test_data
    elif HYPER.PRED_TYPE_ACT_LRN=='temporal':
        testing_data = temporal_test_data
    elif HYPER.PRED_TYPE_ACT_LRN=='spatio-temporal':
        testing_data = spatemp_test_data
        
    return (
        training_data,
        validation_data,
        testing_data
    )


def standardize_features(
    HYPER, 
    raw_data, 
    dataset, 
    reference_data, 
    silent=False
):

    """ Converts the population of each feature into a standard score using mean 
    and std deviations. For X_st, the past time steps of each meteorological 
    condition are transformed separately. For X_s1, the histogram or average 
    values of each channel are transformed separately.
    """

    if HYPER.STANDARDIZATION:

        if not silent:
            # tell us what we do
            print('Standardizing data')

        # get StandardScaler from the sklearn preprocessing package
        standard_scaler = preprocessing.StandardScaler()

        # standardize X_t in the case that it is not OHE
        if HYPER.TIME_ENCODING != 'OHE':
            standard_scaler.fit(reference_data.X_t)
            dataset.X_t = standard_scaler.transform(dataset.X_t)

        # standardize X_st
        for i in range(len(HYPER.METEO_TYPES)):
            standard_scaler.fit(reference_data.X_st[:, :, i])
            dataset.X_st[:, :, i] = standard_scaler.transform(
                dataset.X_st[:, :, i]
            )

        # standardize X_s1
        if HYPER.SPATIAL_FEATURES != 'image':

            for channel in range(raw_data.n_channels):
                standard_scaler.fit(reference_data.X_s1[:, :, channel])
                dataset.X_s1[:, :, channel] = standard_scaler.transform(
                    dataset.X_s1[:, :, channel]
                )

    return dataset
