import math
import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import random

from matplotlib.lines import Line2D

from data import Dataset
from prediction import train_model, test_model
from prediction import load_encoder_and_predictor_weights
from prediction import initialize_optimizer


from sklearn.preprocessing import OrdinalEncoder

class ActLrnResults:

    """ Bundles AL results. """

    def __init__(
        self,
        train_loss,
        val_loss,
        test_loss,
        iter_usage,
        iter_time,
        budget_usage,
        sensor_usage,
        streamtime_usage,
        prediction_model,
        test_data,
        picked_cand_index_set,
        picked_times_index_hist,
        picked_spaces_index_hist,
        picked_inf_score_hist,
        budget_usage_hist,
        iter_time_hist,
        sensor_usage_hist,
        streamtime_usage_hist,
        val_loss_hist,
        initial_sensors_list
    ):

        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        self.iter_usage = iter_usage
        self.iter_time = iter_time
        self.budget_usage = budget_usage
        self.sensor_usage = sensor_usage
        self.streamtime_usage = streamtime_usage
        self.prediction_model = prediction_model
        self.test_data = test_data
        self.picked_cand_index_set = picked_cand_index_set
        self.picked_times_index_hist = picked_times_index_hist
        self.picked_spaces_index_hist = picked_spaces_index_hist
        self.picked_inf_score_hist = picked_inf_score_hist
        self.budget_usage_hist = budget_usage_hist
        self.iter_time_hist = iter_time_hist
        self.sensor_usage_hist = sensor_usage_hist
        self.streamtime_usage_hist = streamtime_usage_hist
        self.val_loss_hist = val_loss_hist
        self.initial_sensors_list = initial_sensors_list


def encode_features(
    HYPER, 
    raw_data, 
    models, 
    dataset,
    available_index_set_update,
    AL_variable, 
    silent=True
):

    """ Encodes features AL_variable of dataset, or passes labels for
    AL_variable being Y_(t,s). Also returns the random index array that is 
    created when CAND_SUBSAMPLE_ACT_LRN is not None and smaller than dataset 
    size. 
    """

    if not silent:
    
        # tell us what we are doing
        print(
            'Encoding features into embedded vector spaces for', AL_variable
        )


    ### Create random subsample before encoding if wanted ###

    # create an index array in the length of the passed dataset
    n_datapoints = len(available_index_set_update)
    index_array = list(available_index_set_update)

    # if we chose a subset of candidate data points, create a random sub-sample
    if (
        HYPER.CAND_SUBSAMPLE_ACT_LRN is not None and 
        HYPER.CAND_SUBSAMPLE_ACT_LRN * dataset.n_datapoints < n_datapoints
    ):

        n_datapoints = math.floor(HYPER.CAND_SUBSAMPLE_ACT_LRN * dataset.n_datapoints)
        index_array = random.sample(
            available_index_set_update, 
            n_datapoints
        )

    # create copy of dataset
    X_t = dataset.X_t[index_array]
    X_s = dataset.X_s[index_array]
    X_st = dataset.X_st[index_array]
    Y = dataset.Y[index_array]

    if HYPER.SPATIAL_FEATURES != 'image':
        X_s1 = dataset.X_s1[index_array]


    ### Encode features here ###

    if AL_variable == 'X_t':
        encoding = models.X_t_encoder.predict(X_t)
        
    elif AL_variable == 'X_st':
        encoding = models.X_st_encoder.predict(X_st)
        
    elif AL_variable == 'X_s1':
    
        if HYPER.SPATIAL_FEATURES != 'image':
            encoding = models.X_s1_encoder.predict(X_s1)

        else:
            ### Encode X_s1 ###
            encoding = np.zeros((n_datapoints, HYPER.ENCODING_NODES_X_s))
            
            # iterate over all datapoints
            for i in range(n_datapoints):
                building_id = X_s[i][0]
                
                # prepare imagery data
                x_s1 = raw_data.building_imagery_data_list[
                    raw_data.building_imagery_id_list.index(int(building_id))
                ]
                x_s1 = np.expand_dims(x_s1, axis=0)
                
                # make predictions and save results in respective matrix
                encoding[i] = models.X_s1_encoder.predict(x_s1)

    elif AL_variable == 'X_(t,s)':

        if HYPER.SPATIAL_FEATURES != 'image':
            encoding = models.X_joint_encoder.predict([X_t, X_s1, X_st])

        else:

            ### Encode X_joint ###
            encoding = np.zeros((n_datapoints, HYPER.ENCODING_NODES_X_joint))

            # iterate over all datapoints
            for i in range(n_datapoints):

                # Get training data of currently iterated batch
                x_t = X_t[i]
                x_st = X_st[i]
                y = Y[i]
                building_id = X_s[i][0]
                cluster_id = X_s[i][1]

                # prepare imagery data
                x_s1 = raw_data.building_imagery_data_list[
                    raw_data.building_imagery_id_list.index(int(building_id))
                ]

                # Expand dimensions for batching
                x_t = np.expand_dims(x_t, axis=0)
                x_s1 = np.expand_dims(x_s1, axis=0)
                x_st = np.expand_dims(x_st, axis=0)

                # Create model input list
                model_input_list = [x_t, x_s1, x_st]

                # make predictions and save results in respective matrix
                encoding[i] = models.X_joint_encoder.predict(model_input_list)

    elif AL_variable == 'Y_hat_(t,s)':

        if HYPER.SPATIAL_FEATURES != 'image':
            encoding = models.prediction_model.predict([X_t, X_s1, X_st])

        else:

            ### Predict Y ###
            encoding = np.zeros((n_datapoints, HYPER.PREDICTION_WINDOW))

            # iterate over all datapoints
            for i in range(n_datapoints):

                # Get training data of currently iterated batch 
                x_t = X_t[i]
                x_st = X_st[i]
                y = Y[i]
                building_id = X_s[i][0]
                cluster_id = X_s[i][1]

                # Prepare imagery data
                x_s1 = raw_data.building_imagery_data_list[
                    raw_data.building_imagery_id_list.index(int(building_id))
                ]

                # Expand dimensions for batching
                x_t = np.expand_dims(x_t, axis=0)
                x_s1 = np.expand_dims(x_s1, axis=0)
                x_st = np.expand_dims(x_st, axis=0)

                # Create model input list
                model_input_list = [x_t, x_s1, x_st]

                # make predictions and save results in respective matrix
                encoding[i] = models.prediction_model.predict(model_input_list)

    elif AL_variable == 'Y_(t,s)':
        encoding = Y
    else:
        print('query variable not recognized.')

    return encoding, index_array


def compute_clusters(
    HYPER, 
    encoding, 
    data_budget_per_iter, 
    silent=True
):

    """ Calculates clusters in the passed encoding vectors using
    HYPER.METHOD_CLUSTERS[0]. Returns cluster labels and centers.
    """

    if not silent:
        # tell us what we are doing
        print(
            'Creating clusters in encodings with n_clusters=', data_budget_per_iter
        )

    # set the clustering method that we chose
    method = HYPER.METHOD_CLUSTERS[0]
    
    # calculate number of clusters
    n_clusters = math.floor(
        data_budget_per_iter * HYPER.POINTS_PER_CLUSTER_ACT_LRN
    )

    # set number of clusters equal to passed or corrected value
    clustering_method = method(n_clusters=n_clusters)

    # cluster encodings
    clustering_method.fit(encoding)
    cluster_labels = clustering_method.labels_
    cluster_centers = clustering_method.cluster_centers_
    
    # get ordinal encoder from Sklearn
    enc = OrdinalEncoder()
    
    # encode labels. NOTE: ordinally encoding clusters ensures that cluster
    # labels start at 0 and end at number of clusters, which is not the case
    # for X_t and X_s1 when not ordinally encoding.
    cluster_labels = enc.fit_transform(
        np.expand_dims(cluster_labels, 1)
    ).astype(int)
    
    cluster_centers = cluster_centers[enc.categories_[0]]
    
    # delete expanded dimension again as it is redundant
    cluster_labels = cluster_labels[:, 0]
    
    # calculate number of clusters created in data
    n_clusters = max(cluster_labels) + 1
    
    return cluster_labels, cluster_centers, n_clusters


def compute_similarity(
    HYPER, 
    encoding, 
    cluster_labels, 
    cluster_centers, 
    silent=True
):

    """ Calculates distances to cluster centers. A large value means that 
    encoding is close to its cluster center. A small value means that encoding 
    is far from cluster center.
    """

    if not silent:
        # tell us what we are doing
        print("Calculating distances" )

        # create a progress bar for training
        progbar_distance = tf.keras.utils.Progbar(len(encoding))

    # get the kernel function we chose
    metric = HYPER.METRIC_DISTANCES[0]
    
    # set the number of encoded data points
    n_enc_datapoints = len(encoding)
    
    # CAUTION: create shape (n_enc_datapoints,) instead of (n_enc_datapoints, 1)
    similarity_array = np.zeros((n_enc_datapoints,))

    # iterate over all encodings
    for i in range(n_enc_datapoints):
        
        # get encoding's cluster label
        label = cluster_labels[i]
        
        # get cluster's center
        center = cluster_centers[label]
        
        # calculate similarity/closeness of encoding to its cluster center
        similarity_array[i] = metric(
            np.expand_dims(center, axis=0), np.expand_dims(encoding[i], axis=0)
        )
        
        if not silent:
            # increment progress bar
            progbar_distance.add(1)

    return similarity_array


def feature_embedding_AL(
    HYPER,
    pred_type,
    models,
    raw_data,
    train_data,
    candidate_dataset,
    loss_object,
    optimizer,
    mean_loss,
    loss_function,
    method,
    AL_variable=None,
    silent=True,
):

    """ Given the prediction models 'models' which are trained on the initially 
    available data points 'train_data', it selects a batch of data points to 
    query labels for from the pool candidate data points 'candidate_dataset'. 
    Three different methods can be chosen through 'method' and set to 
    'max d_c', 'min d_c' and 'rnd d_c', each standing for another 
    variant of the algorithm:
        1. 'max d_c': maximizes embedding uncertainty
        2. 'min d_c': minimizes embedding uncertainty
        3. 'rnd d_c': randomizes embedding uncertainty from each cluster uniformly
    """

    ### Compute some initial values ###

    # compute total data budget
    data_budget_total = math.floor(
        HYPER.DATA_BUDGET_ACT_LRN * candidate_dataset.n_datapoints
    )
    
    # compute data budget available in each query iteration
    data_budget_per_iter = math.floor(
        data_budget_total / HYPER.N_ITER_ACT_LRN
    )

    # compute number of sensors and times in initial training data
    n_times_0 = len(np.unique(train_data.X_t, axis=0))
    n_sensors_0 = len(np.unique(train_data.X_s, axis=0))
    
    # compute number of new times in candidate data    
    n_times_new = (
        len(
            np.unique(
                np.concatenate(
                    (train_data.X_t, candidate_dataset.X_t)
                ), 
                axis=0)
            )
        - n_times_0
    )
    
    # compute number of new sensors in candidate data 
    n_sensors_new = (
        len(
            np.unique(
                np.concatenate(
                    (train_data.X_s, candidate_dataset.X_s)
                ), 
                axis=0)
            )
        - n_sensors_0
    )
    
    # create a list of initial sensors for visualizing data selection maps later
    initial_sensors_list = list(set(train_data.X_s[:, 0]))

    if not silent:
        # tell us what we are doing
        print(
            'prediction task:             {}'.format(
                pred_type
            )
        )
        
        print(
            'AL variable:                 {}'.format(
                AL_variable
            )
        )
        
        print(
            'AL variant:                  {}'.format(
                method
            )
        )
        
        print(
            'distance metric:             {}'.format(
                HYPER.DISTANCE_METRIC_ACT_LRN
            )
        )
        
        print(
            'clustering method:           {}'.format(
                HYPER.CLUSTER_METHOD_ACT_LRN
            )
        )
        
        print(
            'data budget:                 {}/{} ({:.0%})'.format(
                data_budget_total, 
                candidate_dataset.n_datapoints, 
                HYPER.DATA_BUDGET_ACT_LRN
            )
        )
        
        print(
            'known sensors:               {}'.format(
                n_sensors_0
            )
        )
        
        print(
            'known streaming timestamps:  {}'.format(
                n_times_0
            )
        )
        
        print(
            'candidate sensors:           {}'.format(
                n_sensors_new
            )
        )
        
        print(
            'candidate timestamps:        {}'.format(
                n_times_new
            )
        )


    ### Load model weights ###

    # Note: if you load entire initial models, instead of their weights only,
    # network configuration information is lost and tf will not train encoders
    # alongside training the main prediction model.
    models = load_encoder_and_predictor_weights(
        raw_data, 
        models, 
        pred_type
    )
    
    (
        loss_object, 
        optimizer, 
        loss_function, 
        mean_loss,
    ) = initialize_optimizer(HYPER)


    ### Start AL algorithm ###

    # initialize some counters
    data_counter = 0
    sensor_counter = 0
    streamtime_counter = 0
    picked_cand_index_set = set()
    available_index_set_update = set(np.arange(candidate_dataset.n_datapoints))
    budget_usage_hist = []
    iter_time_hist = []
    picked_times_index_hist = []
    picked_spaces_index_hist = []
    picked_inf_score_hist = []
    sensor_usage_hist = []
    streamtime_usage_hist = []
    val_loss_hist = []
    
    # Set starting time of algorithm
    t_start_0 = timeit.default_timer()

    # start Active Learning iterations
    for iteration in range(HYPER.N_ITER_ACT_LRN):

        if not silent:
            # mark beginning of iteration
            print('---' * 3)

        # Set the start time
        t_start = timeit.default_timer()
        
        # calculate candidate subsample size to compare vs batch query size
        if HYPER.CAND_SUBSAMPLE_ACT_LRN is not None:
            subsample_size = math.floor(
                candidate_dataset.n_datapoints 
                * HYPER.CAND_SUBSAMPLE_ACT_LRN 
            )
        
        ### Choose candidates to query ###
  
        if method == 'PL':
            ### Choose queries according to PL (random) *tested* ###

            # Create a random batch_index_array
            batch_index_list = random.sample(
                available_index_set_update, 
                data_budget_per_iter
            )

        elif HYPER.CAND_SUBSAMPLE_ACT_LRN is not None and subsample_size <= data_budget_per_iter:
           ### AL Exception: choose queries at random *tested* ###

            # tell us what is going on
            if not silent:
                print('Attention! Candidate subsample is smaller than query batch')
                print('subsample size:', subsample_size)
                print('query batch size:', data_budget_per_iter)
            
            # set batch size equal to subsample size for printing later if not silent
            data_budget_per_iter = int(subsample_size)
            
            # create a list of length data_budget_per_iter with zero information scores 
            inf_score_list = [0] * data_budget_per_iter
            
            # Create a random batch_index_array
            batch_index_list = random.sample(
                available_index_set_update, 
                data_budget_per_iter
            )
        
        else:
            ### Encode data points *tested* ###

            candidate_encoded, cand_sub_index = encode_features(
                HYPER,
                raw_data,
                models,
                candidate_dataset,
                available_index_set_update,
                AL_variable,
            )
            
            ### Calculate clusters *tested* ###

            cand_labels, cand_centers, n_clusters = compute_clusters(
                HYPER, 
                candidate_encoded, 
                data_budget_per_iter
            )
            
            ### Compute similarity values for each candidate ###
            
            if method != 'rnd d_c':
                ### Calculate distances *tested* ###

                # calculates far points with small similarity value
                cand_similarity_array = compute_similarity(
                    HYPER, 
                    candidate_encoded, 
                    cand_labels, 
                    cand_centers
                )
                
                if method == 'max d_c':
                    # reverse order by multiplying with -1 
                    # --> smallest becomes most similar
                    # --> turns similarity into distance array
                    cand_similarity_array = -1 * cand_similarity_array

            
            ### Choose data from clusters *tested* ###
            
            # create zero array that is filled with cluster IDs for this batch
            batch_index_list = []
            inf_score_list = []

            # iterates over the batch_index_array up to data_budget_per_iter
            cluster_batch_counter = 0
            
            # iterates over clusters until n_clusters, then resets to 0
            # if cluster_batch_counter does not reached data_budget_per_iter
            cluster_index = 0
            
            # iterate over all clusters until cluster_batch_counter reaches 
            # data_budget_per_iter
            while cluster_batch_counter < data_budget_per_iter:

                # get an array of indices matching to currently iterated cluster 
                # ID
                index_array = np.where(cand_labels == cluster_index)[0]

                # if the set is not empty
                if len(index_array) != 0:
                
                    if method == 'rnd d_c':
                        # choose one element at random from this index array
                        index_choice = np.random.choice(index_array)
                        
                    else:
                        # get similarity values for matching index array
                        similarity_array = cand_similarity_array[index_array]

                        if method == 'avg d_c':
                            # turn into absolute difference to average similarity
                            similarity_array = abs(
                                similarity_array - np.mean(similarity_array)
                            )

                        # calculate largest similarity
                        max_similarity = similarity_array.max()
                        
                        # choose first/largest value from similarity_array
                        index_choice = index_array[
                            np.where(
                                similarity_array == max_similarity
                            )[0][0]
                        ]
                        
                        # add information content score of data point to inf_score_list
                        inf_score_list.append(max_similarity)

                    # add randomly chosen index to zero array
                    batch_index_list.append(cand_sub_index[index_choice])

                    # setting the cluster ID to -1 excludes data point from  
		                # considerations in next iterations of this loop
                    cand_labels[index_choice] = -1

                    # increment the counter for already added data points to
                    # zero array
                    cluster_batch_counter += 1

                # increment the cluster ID index for the next iteration
                cluster_index += 1

                # set cluster ID index to zero for next iteration if an entire 
	              # round of iterations did not fill zero array
                if cluster_index >= n_clusters:
                    cluster_index = 0
    
    
        ### Compute the set of queried data points *tested* ###
        
        # compute how many points were queried until last iteration
        n_used_cand_data_total = len(picked_cand_index_set)
        
        # update the set of points queried until now including this iteration
        picked_cand_index_set = picked_cand_index_set.union(
            set(batch_index_list)
        )
        
        # create a list from the set
        picked_cand_index_list = list(picked_cand_index_set)
        
        # compute the number of new data points queried in this iteration
        n_new_data = len(picked_cand_index_set) - n_used_cand_data_total
        

        ### Create new training batch ###
        
        if HYPER.EXTEND_TRAIN_DATA_ACT_LRN:
            # get share of training data from the pool of possible testing points
            X_t_ord_1D_new_train = np.concatenate(
                (
                    train_data.X_t_ord_1D, 
                    candidate_dataset.X_t_ord_1D[
                        picked_cand_index_list
                    ]
                ), 
                axis=0
            )
            
            X_t_new_train = np.concatenate(
                (
                    train_data.X_t, 
                    candidate_dataset.X_t[
                        picked_cand_index_list
                    ]
                ), 
                axis=0
            )
            
      
            X_s_new_train = np.concatenate(
                (
                    train_data.X_s, 
                    candidate_dataset.X_s[
                        picked_cand_index_list
                    ]
                ), 
                axis=0
            )
            
            X_st_new_train = np.concatenate(
                (
                    train_data.X_st, 
                    candidate_dataset.X_st[
                        picked_cand_index_list
                    ]
                ),
                axis=0
            )
            
            Y_new_train = np.concatenate(
                (
                    train_data.Y, 
                    candidate_dataset.Y[
                        picked_cand_index_list
                    ]
                ), 
                axis=0
            )

            if HYPER.SPATIAL_FEATURES != 'image':
                X_s1_new_train = np.concatenate(
                    (
                        train_data.X_s1, 
                        candidate_dataset.X_s1[
                            picked_cand_index_list
                        ]
                    ), 
                    axis=0
                )

            else:
                X_s1_new_train = 0
        
        else:
            # get share of training data from pool of possible testing points
            X_t_ord_1D_new_train = candidate_dataset.X_t_ord_1D[batch_index_list]
            X_t_new_train = candidate_dataset.X_t[batch_index_list]
            X_s_new_train = candidate_dataset.X_s[batch_index_list]
            X_st_new_train = candidate_dataset.X_st[batch_index_list]
            Y_new_train = candidate_dataset.Y[batch_index_list]

            if HYPER.SPATIAL_FEATURES != 'image':
                X_s1_new_train = candidate_dataset.X_s1[
                    batch_index_list
                ]

            else:
                X_s1_new_train = 0


        ### Update training data for counting sensors and stream times ###
        
        # causing duplicate points on purpose when RED_CAND_DATA_ACT_LRN=False
        train_data_update_X_t_ord_1D = np.concatenate(
            (
                train_data.X_t_ord_1D, 
                candidate_dataset.X_t_ord_1D[
                    picked_cand_index_list
                ]
            ), 
            axis=0
        )
        
        train_data_update_X_t = np.concatenate(
            (
                train_data.X_t, 
                candidate_dataset.X_t[
                    picked_cand_index_list
                ]
            ), 
            axis=0
        )
        
        train_data_update_X_s = np.concatenate(
            (
                train_data.X_s, 
                candidate_dataset.X_s[
                    picked_cand_index_list
                ]
            ), 
            axis=0
        )
        
        
        ### Update candidate data ###

        # update candidate data if chosen so
        if HYPER.RED_CAND_DATA_ACT_LRN:
            # update set of available indices
            available_index_set_update = (
                available_index_set_update - picked_cand_index_set
            )


        ### Create (updated) validation data ###

        # update validation data if chosen so
        if HYPER.UPD_VAL_DATA_ACT_LRN:
            X_t_ord_1D_new_val = np.delete(
                candidate_dataset.X_t_ord_1D, picked_cand_index_list, 0
            )
            
            X_t_new_val = np.delete(
                candidate_dataset.X_t, picked_cand_index_list, 0
            )
            
            X_s_new_val = np.delete(
                candidate_dataset.X_s, picked_cand_index_list, 0
            )
            
            X_st_new_val = np.delete(
                candidate_dataset.X_st, picked_cand_index_list, 0
            )
            
            Y_new_val = np.delete(
                candidate_dataset.Y, picked_cand_index_list, 0
            )

            if HYPER.SPATIAL_FEATURES != 'image':
                X_s1_new_val = np.delete(
                    candidate_dataset.X_s1, picked_cand_index_list, 0
                )
            else:
                X_s1_new_val = 0

            # update for controlling subsampling size vs. query batch size
            # and for creating the right subsample size during feature encoding
            candidate_dataset.n_datapoints = len(candidate_dataset.X_t_ord_1D)

        else:
            # create new validation data by copying from initial candidate data
            X_t_ord_1D_new_val = candidate_dataset.X_t_ord_1D
            X_t_new_val = candidate_dataset.X_t
            X_s_new_val = candidate_dataset.X_s
            X_st_new_val = candidate_dataset.X_st
            Y_new_val = candidate_dataset.Y

            if HYPER.SPATIAL_FEATURES != 'image':
                X_s1_new_val = candidate_dataset.X_s1
            else:
                X_s1_new_val = 0


        ### Train and validate with new batches, avoids unwanted shuffling ###

        # create new training dataset
        new_train_batch = Dataset(
            X_t_ord_1D_new_train,
            X_t_new_train, 
            X_s_new_train, 
            X_s1_new_train, 
            X_st_new_train, 
            Y_new_train
        )

        # create new validation dataset
        new_val_data = Dataset(
            X_t_ord_1D_new_val,
            X_t_new_val, 
            X_s_new_val, 
            X_s1_new_val, 
            X_st_new_val, 
            Y_new_val
        )

        # train model with new data
        train_hist_batch, val_hist_batch = train_model(
            HYPER,
            models.prediction_model,
            new_train_batch,
            new_val_data,
            raw_data,
            loss_object,
            optimizer,
            mean_loss,
        )
        
        # keep track of loss histories
        if iteration == 0:
            train_hist = train_hist_batch
            val_hist = val_hist_batch
        else:
            train_hist = np.concatenate((train_hist, train_hist_batch))
            val_hist = np.concatenate((val_hist, val_hist_batch))


        ### Update counters ###

        # get ending time
        t_end = timeit.default_timer()

        # increment data cointer
        data_counter += n_new_data
        
        # increment sensor counter
        sensor_counter = len(
            np.unique(train_data_update_X_s, axis=0)
        ) - n_sensors_0
        
        # increment sensor counter
        streamtime_counter = len(
            np.unique(train_data_update_X_t, axis=0)
        ) - n_times_0

        # budget share that is eventually used
        cand_data_usage = data_counter / data_budget_total

        # time in seconds that is used in this iteration
        iter_time = math.ceil(t_end - t_start)
        
        # if there were any new sensors to add, get share that was added
        if n_sensors_new != 0:
            percent_sensors = sensor_counter / n_sensors_new
        else:
            percent_sensors = 0

        # if there were any new streamtimes to add, get share that was added
        if n_times_new != 0:
            percent_streamtimes = streamtime_counter / n_times_new
        else:
            percent_streamtimes = 0
            
        # add data usage to history
        budget_usage_hist.append(cand_data_usage)
        
        # add iteration time to history
        iter_time_hist.append(iter_time)
        
        # add sensor usage to history
        sensor_usage_hist.append(percent_sensors)
        
        # add streamtime usage to history
        streamtime_usage_hist.append(percent_streamtimes)
        
        # add batch index times to history
        picked_times_index_hist.append(candidate_dataset.X_t_ord_1D[batch_index_list])
        
        # add batch index spaces to history
        picked_spaces_index_hist.append(candidate_dataset.X_s[batch_index_list][:, 0])
        
        # add similarity scores to history if not 'PL' and 'rnd d_c' methods
        if method != 'PL' and method != 'rnd d_c':
            # if 'max d_c' method then the information score is negative and reversed
            # so adding with one equals 1 - (- similarity) and gives information score
            if method == 'max d_c':
                picked_inf_score_hist.append([1+x for x in inf_score_list])
            else:
                picked_inf_score_hist.append(inf_score_list)
                
        # add last validation loss value to test loss history
        val_loss_hist.append(val_hist[-1])

        if not silent:
            # tell us the numbers
            print(
                'Iteration:                            {}'.format(
                    iteration
                )
            )
            
            print(
                'Time:                                 {}s'.format(
                    iter_time
                )
            )
            
            print(
                'Trained on candidate batch size:      {}'.format(
                    data_budget_per_iter
                )
            )
            
            print(
                'Used streaming times:                 {}/{} ({:.0%})'.format(
                    streamtime_counter, n_times_new, percent_streamtimes
                )
            )
            
            print(
                'Used sensors:                         {}/{} ({:.0%})'.format(
                    sensor_counter, n_sensors_new, percent_sensors
                )
            )
            
            print(
                'Used data budget:                     {}/{} ({:.0%})'.format(
                    data_counter, data_budget_total, cand_data_usage
                )
            )

    # mark end of test for currently iterated sorting array
    if not silent:
        print('---' * 20)

    # time in seconds that is eventually used
    iter_time = math.ceil(t_end - t_start_0)


    ### Create test dataset and predict ###

    # create new validation data by deleting batch of picked data from candidates
    X_t_ord_1D_test = np.delete(
        candidate_dataset.X_t_ord_1D, 
        picked_cand_index_list, 
        0
    )
    
    X_t_test = np.delete(
        candidate_dataset.X_t, 
        picked_cand_index_list, 
        0
    )
    
    X_s_test = np.delete(
        candidate_dataset.X_s, 
        picked_cand_index_list, 
        0
    )
    
    X_st_test = np.delete(
        candidate_dataset.X_st, 
        picked_cand_index_list, 
        0
    )
    
    Y_test = np.delete(
        candidate_dataset.Y, 
        picked_cand_index_list, 
        0
    )

    if HYPER.SPATIAL_FEATURES != 'image':
        X_s1_test = np.delete(
            candidate_dataset.X_s1, 
            picked_cand_index_list, 
            0
        )
        
    else:
        X_s1_test = 0

    # create a copy of candidate test data
    test_data = Dataset(
        X_t_ord_1D_test,
        X_t_test, 
        X_s_test, 
        X_s1_test, 
        X_st_test, 
        Y_test
    )

    # Predict on candidate datapoints that are not in training data
    title = '{} {} {}'.format(pred_type, AL_variable, method)
    
    test_loss = test_model(
        HYPER, 
        title, 
        models.prediction_model, 
        test_data, 
        raw_data, 
        mean_loss, 
        loss_function
    )


    ### Shorten test dataset to random subsample ###
    
    if HYPER.SAVED_SAMPLES_ACT_LRN >= test_data.n_datapoints:
        rnd_array = np.arange(test_data.n_datapoints)
    else:
        # choose a subsample of the test data for saving
        rnd_array = random.sample(
            list(np.arange(test_data.n_datapoints)), 
            HYPER.SAVED_SAMPLES_ACT_LRN
        )

    X_t_ord_1D_test = X_t_ord_1D_test[rnd_array]
    X_t_test = X_t_test[rnd_array]
    X_s_test = X_s_test[rnd_array]
    X_st_test = X_st_test[rnd_array]
    Y_test = Y_test[rnd_array]

    if HYPER.SPATIAL_FEATURES != 'image':
        X_s1_test = X_s1_test[rnd_array]
    else:
        X_s1_test = 0

    # overwrite test_data with samples you want to save
    test_data = Dataset(
        X_t_ord_1D_test,
        X_t_test, 
        X_s_test, 
        X_s1_test, 
        X_st_test, 
        Y_test
    )

    
    ### Create a results object ###

    # create an ActLrnResults object and pass the results for compactness
    results = ActLrnResults(
        train_hist,
        val_hist,
        test_loss,
        HYPER.N_ITER_ACT_LRN,
        iter_time,
        cand_data_usage,
        percent_sensors,
        percent_streamtimes,
        models.prediction_model,
        test_data,
        picked_cand_index_set,
        picked_times_index_hist,
        picked_spaces_index_hist,
        picked_inf_score_hist,
        budget_usage_hist,
        iter_time_hist,
        sensor_usage_hist,
        streamtime_usage_hist,
        val_loss_hist,
        initial_sensors_list
    )

    return results


def test_AL_sequence_importance(
    HYPER,
    pred_type,
    models,
    raw_data,
    train_data,
    candidate_dataset,
    loss_object,
    optimizer,
    mean_loss,
    loss_function,
    AL_results,
    method,
    AL_variable=None,
    silent=True
):
    
    """ Tests the importance of the query sequence for passed AL results """

    if HYPER.TEST_SEQUENCE_IMPORTANCE:
        if not silent:
            # create a progress bar for training
            progbar_seqimportance = tf.keras.utils.Progbar(AL_results.iter_usage)

            # tell us what we are doing
            print('Testing sequence importance for')
            
            print(
                'prediction type:                      {}'.format(
                    pred_type
                )
            )
            
            print(
                'query variable:                       {}'.format(
                    AL_variable
                )
            )
            
            print(
                'query variant:                        {}'.format(
                    method
                )
            )
        

        ### Load model weights ###

        # Note: if you load entire initial models, instead of their weights only,
        # network configuration information is lost and tf will not train encoders
        # alongside training the main prediction model.
        models = load_encoder_and_predictor_weights(
            raw_data, 
            models, 
            pred_type
        )

        ### Start AL algorithm with random sequence selection ###

        # initialize some values
        data_budget_total = math.floor(
            HYPER.DATA_BUDGET_ACT_LRN * candidate_dataset.n_datapoints
        )
        # compute data budget available in each query iteration
        data_budget_per_iter = math.floor(
            data_budget_total / HYPER.N_ITER_ACT_LRN
        )
        picked_cand_index_set = set()
        available_index_set_update = AL_results.picked_cand_index_set
        data_counter = 0
        
        # start AL iterations
        for iteration in range(AL_results.iter_usage):

            ### Choose training batch ###

            # Create a random splitting array
            batch_index_list = random.sample(
                available_index_set_update, 
                data_budget_per_iter
            )
            
            # update candidate indices and data counter
            picked_cand_index_set = picked_cand_index_set.union(
                set(batch_index_list)
            )
            picked_cand_index_list = list(picked_cand_index_set)
            data_counter = len(picked_cand_index_list)


            ### Create training data ####
            
            if HYPER.EXTEND_TRAIN_DATA_ACT_LRN:
                # get share of training data from pool of possible testing points
                X_t_ord_1D_new_train = np.concatenate(
                    (
                        train_data.X_t_ord_1D, 
                        candidate_dataset.X_t_ord_1D[picked_cand_index_list]
                    ), 
                    axis=0
                )
                X_t_new_train = np.concatenate(
                    (
                        train_data.X_t, 
                        candidate_dataset.X_t[picked_cand_index_list]
                    ), 
                    axis=0
                )
                X_s_new_train = np.concatenate(
                    (
                        train_data.X_s, 
                        candidate_dataset.X_s[picked_cand_index_list]
                    ), 
                    axis=0
                )
                X_st_new_train = np.concatenate(
                    (
                        train_data.X_st, 
                        candidate_dataset.X_st[picked_cand_index_list]
                    ), 
                    axis=0
                )
                Y_new_train = np.concatenate(
                    (
                        train_data.Y, 
                        candidate_dataset.Y[picked_cand_index_list]
                    ), 
                    axis=0
                )

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_train = np.concatenate(
                        (
                            train_data.X_s1, 
                            candidate_dataset.X_s1[picked_cand_index_list]
                        ), 
                        axis=0
                    )
                else:
                    X_s1_new_train = 0
            
            else:
                # sort all initial candidate data features with the same array
                X_t_ord_1D_new_train = candidate_dataset.X_t_ord_1D[batch_index_list]
                X_t_new_train = candidate_dataset.X_t[batch_index_list]
                X_s_new_train = candidate_dataset.X_s[batch_index_list]
                X_st_new_train = candidate_dataset.X_st[batch_index_list]
                Y_new_train = candidate_dataset.Y[batch_index_list]

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_train = candidate_dataset.X_s1[batch_index_list]
                else:
                    X_s1_new_train = 0
                    

            ### Update picked_cand_index_list ###

            # update candidate data if chosen so
            if HYPER.RED_CAND_DATA_ACT_LRN:
                # update set of available indices
                available_index_set_update = (
                    available_index_set_update - picked_cand_index_set
                )

            ### Create (updated) validation data ###
            
            if HYPER.UPD_VAL_DATA_ACT_LRN:
                # create new validation data by deleting the batch
                X_t_ord_1D_new_val = np.delete(
                    candidate_dataset.X_t_ord_1D, picked_cand_index_list, 0
                )
                X_t_new_val = np.delete(
                    candidate_dataset.X_t, picked_cand_index_list, 0
                )
                X_s_new_val = np.delete(
                    candidate_dataset.X_s, picked_cand_index_list, 0
                )
                X_st_new_val = np.delete(
                    candidate_dataset.X_st, picked_cand_index_list, 0
                )
                Y_new_val = np.delete(
                    candidate_dataset.Y, picked_cand_index_list, 0
                )

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_val = np.delete(
                        candidate_dataset.X_s1, picked_cand_index_list, 0
                    )
                else:
                    X_s1_new_val = 0

            else:
                # create new validation data by copying initial candidates
                X_t_ord_1D_new_val = candidate_dataset.X_t_ord_1D
                X_t_new_val = candidate_dataset.X_t
                X_s_new_val = candidate_dataset.X_s
                X_st_new_val = candidate_dataset.X_st
                Y_new_val = candidate_dataset.Y

                if HYPER.SPATIAL_FEATURES != 'image':
                    X_s1_new_val = candidate_dataset.X_s1
                else:
                    X_s1_new_val = 0


            ### Train with new batch ###

            # bundle chosen batch of candidate data points as Dataset object
            new_train_batch = Dataset(
                X_t_ord_1D_new_train,
                X_t_new_train, 
                X_s_new_train, 
                X_s1_new_train, 
                X_st_new_train, 
                Y_new_train
            )

            # bundle updated data points as Dataset object for validation. This 
            # avoids unwanted shuffling
            new_val_data = Dataset(
                X_t_ord_1D_new_val,
                X_t_new_val, 
                X_s_new_val, 
                X_s1_new_val, 
                X_st_new_val, 
                Y_new_val
            )

            # train model with new data
            train_hist_batch, val_hist_batch = train_model(
                HYPER,
                models.prediction_model,
                new_train_batch,
                new_val_data,
                raw_data,
                loss_object,
                optimizer,
                mean_loss
            )

            # keep track of loss histories
            if iteration == 0:
                train_hist = train_hist_batch
                val_hist = val_hist_batch
            else:
                train_hist = np.concatenate((train_hist, train_hist_batch))
                val_hist = np.concatenate((val_hist, val_hist_batch))

            if not silent:
                # increment progress bar
                progbar_seqimportance.add(1)


        ### Create test dataset and predict ###

        # create new validation data by deleting the batch of picked data from 
        # candidate dataset
        X_t_ord_1D_test = np.delete(
            candidate_dataset.X_t_ord_1D, 
            picked_cand_index_list, 
            0
        )
        
        X_t_test = np.delete(
            candidate_dataset.X_t, 
            picked_cand_index_list, 
            0
        )
        
        X_s_test = np.delete(
            candidate_dataset.X_s, 
            picked_cand_index_list, 
            0
        )
        
        X_st_test = np.delete(
            candidate_dataset.X_st, 
            picked_cand_index_list, 
            0
        )
        
        Y_test = np.delete(
            candidate_dataset.Y, 
            picked_cand_index_list, 
            0
        )

        if HYPER.SPATIAL_FEATURES != "image":
            X_s1_test = np.delete(
                candidate_dataset.X_s1, 
                picked_cand_index_list, 
                0
            )
        else:
            X_s1_test = 0

        # create a copy of candidate test data
        test_data = Dataset(
            X_t_ord_1D_test,
            X_t_test, 
            X_s_test, 
            X_s1_test, 
            X_st_test, 
            Y_test
        )

        # Predict on candidate datapoints that are not in training data
        title = '{} {} {}'.format(pred_type, AL_variable, method)
        
        test_loss = test_model(
            HYPER, 
            title, 
            models.prediction_model, 
            test_data, 
            raw_data, 
            mean_loss,
            loss_function 
        )

        AL_results.seqimportance_train_loss = train_hist
        AL_results.seqimportance_val_loss = val_hist
        AL_results.seqimportance_test_loss = test_loss

        if not silent: 
            # Indicate termination of execute
            print('---' * 20)

    return AL_results



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
            train_loss = PL_results.train_loss
            val_loss = PL_results.val_loss

            legend_name = ('PL: {}s- {:.0%} budget'
            '- {:.0%} sensors- {:.0%} times- {:.2} loss').format(
                PL_results.iter_time,
                PL_results.budget_usage,
                PL_results.sensor_usage,
                PL_results.streamtime_usage,
                PL_results.test_loss,
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
                    AL_result.iter_time,
                    AL_result.budget_usage,
                    AL_result.sensor_usage,
                    AL_result.streamtime_usage,
                    AL_result.test_loss,
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


def save_act_lrn_models(
    HYPER, 
    raw_data, 
    AL_result_list, 
    PL_result_list
):

    """ Saves the actively trained prediction models. """

    if HYPER.SAVE_ACT_LRN_MODELS:

        for index_pred, pred_type in enumerate(HYPER.PRED_LIST_ACT_LRN):

            # get method_result_list of currently iterated prediction type
            var_result_list = AL_result_list[index_pred]

            # get random results
            PL_results = PL_result_list[index_pred]

            prediction_model = PL_results.prediction_model

            # create the full path for saving random  prediction model
            saving_path = raw_data.path_to_AL_models + pred_type + '/'
        
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)
                
            path_to_model = saving_path + 'PL.h5'

            # save currently iterated model
            prediction_model.save(path_to_model)

            for index_var, AL_variable in enumerate(HYPER.QUERY_VARIABLES_ACT_LRN):

                # get variable result list
                method_result_list = var_result_list[index_var]

                for index_method, method in enumerate(
                    HYPER.QUERY_VARIANTS_ACT_LRN
                ):

                    # get result object and prediction model
                    AL_result = method_result_list[index_method]
                    prediction_model = AL_result.prediction_model

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
    RF_results, 
    AL_result_list, 
    PL_result_list
):

    """ Saves the active learning results, including number of iterations used, 
    time used for each iteration, share of data budget used, share of sensor 
    budget used, share of stream time budget used, testing loss baseline loss 
    and passive learning benchmark histories, validation histories and training 
    histories.
    """

    if HYPER.SAVE_ACT_LRN_RESULTS:

        for pred_index, pred_type in enumerate(HYPER.PRED_LIST_ACT_LRN):
            saving_path = raw_data.path_to_AL_results + pred_type + '/'
            
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)
                
            path_to_results_file = saving_path + 'results.csv'
            
            # create empty DataFrame
            result_df = pd.DataFrame()
            df_list = []

            # baseline results
            RF_loss = RF_results[pred_type]

            # get method_result_list of currently iterated prediction type
            var_result_list = AL_result_list[pred_index]

            ### Save PL results ###
            
            # get PL results
            PL_results = PL_result_list[pred_index]

            n_iterations = PL_results.iter_usage
            t_iterations = PL_results.iter_time
            budget_usage = PL_results.budget_usage
            sensor_usage = PL_results.sensor_usage
            streamtime_usage = PL_results.streamtime_usage
            test_loss = PL_results.test_loss

            train_loss = PL_results.train_loss
            val_loss = PL_results.val_loss

            col_name_train = '{} {} {} train'.format(pred_type, None, 'PL')
            col_name_val = '{} {} {} val'.format(pred_type, None, 'PL')

            meta_entry = np.array(
                [
                    n_iterations,
                    t_iterations,
                    budget_usage,
                    sensor_usage,
                    streamtime_usage,
                    RF_loss,
                    test_loss,
                ]
            )
            entry_train = np.concatenate((meta_entry, train_loss))
            entry_val = np.concatenate((meta_entry, val_loss))

            df_list.append(
                pd.DataFrame({col_name_train: pd.Series(entry_train)})
            )
            df_list.append(
                pd.DataFrame({col_name_val: pd.Series(entry_val)})
            )
            
            ### Prepare sequence imporatance for AL ### 
            if HYPER.TEST_SEQUENCE_IMPORTANCE:
                seqimportance_df = pd.DataFrame()
                df_list_seqimportance = []
                path_to_seqimportance_file = (
                    saving_path 
                    + 'sequence_importance.csv'
                )

            ### Prepare budget vs accuracy for PL ###
            
            path_to_budgetvsaccuracy_file = saving_path + 'budget_vs_accuracy.csv'
            budgetvsaccuracy_df = pd.DataFrame()
            budgetvsaccuracy_df_list = []
            data = np.rint(100 * np.array(PL_results.budget_usage_hist)).astype(int)
            sensors = np.rint(100 * np.array(PL_results.sensor_usage_hist)).astype(int)
            streamtimes = np.rint(100 * np.array(PL_results.streamtime_usage_hist)).astype(int)
            val_loss = PL_results.val_loss_hist
            accuracy = np.rint(100 * (1 - np.minimum(1, val_loss / RF_loss))).astype(int)
            
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

 
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_data: pd.Series(data)})
            )
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_sensors: pd.Series(sensors)})
            )
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_streamtimes: pd.Series(streamtimes)})
            )
            budgetvsaccuracy_df_list.append(
                pd.DataFrame({col_name_accuracy: pd.Series(accuracy)})
            )
                
            ### Prepare picked times and spaces for PL ###
            path_to_spacetime_file = saving_path + 'spacetime_selection.csv'
            spacetime_df = pd.DataFrame()
            spacetime_df_list = []
            picked_times_index_hist = PL_results.picked_times_index_hist
            picked_spaces_index_hist = PL_results.picked_spaces_index_hist
            initial_sensors_list = PL_results.initial_sensors_list
            
            col_name_initial_sensors = '{} - initial sensors'.format(
                pred_type
            )
            
            spacetime_df_list.append(
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
                
                spacetime_df_list.append(
                    pd.DataFrame({col_name_times: pd.Series(picked_times_list)})
                )
                spacetime_df_list.append(
                    pd.DataFrame({col_name_spaces: pd.Series(picked_spaces_list)})
                )
            
            
            for index_var, AL_variable in enumerate(HYPER.QUERY_VARIABLES_ACT_LRN):

                ### Save main AL results ### 
                method_result_list = var_result_list[index_var]

                for index_method, method in enumerate(
                    HYPER.QUERY_VARIANTS_ACT_LRN
                ):

                    AL_result = method_result_list[index_method]

                    n_iterations = AL_result.iter_usage
                    t_iterations = AL_result.iter_time
                    budget_usage = AL_result.budget_usage
                    sensor_usage = AL_result.sensor_usage
                    streamtime_usage = AL_result.streamtime_usage
                    test_loss = AL_result.test_loss
                    delta_loss_RF = AL_result.test_loss - RF_loss
                    delta_loss_PL = (
                        AL_result.test_loss 
                        - PL_results.test_loss
                    )

                    train_loss = AL_result.train_loss
                    val_loss = AL_result.val_loss
                    
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

                    meta_entry = np.array(
                        [
                            n_iterations,
                            t_iterations,
                            budget_usage,
                            sensor_usage,
                            streamtime_usage,
                            RF_loss,
                            test_loss,
                            delta_loss_RF,
                            delta_loss_PL,
                        ]
                    )
                    entry_train = np.concatenate((meta_entry, train_loss))
                    entry_val = np.concatenate((meta_entry, val_loss))

                    df_list.append(
                        pd.DataFrame({col_name_train: pd.Series(entry_train)})
                    )
                    df_list.append(
                        pd.DataFrame({col_name_val: pd.Series(entry_val)})
                    )
                    
                    ### Save sequence importance for AL ### 
                    if HYPER.TEST_SEQUENCE_IMPORTANCE:
                        train_loss_seqimportance = (
                            AL_result.seqimportance_train_loss
                        )
                        val_loss_seqimportance = (
                            AL_result.seqimportance_val_loss
                        )
                        test_loss_seqimportance = (
                            AL_result.seqimportance_test_loss
                        )
                        meta_entry = np.array(
                            [
                                test_loss_seqimportance
                            ]
                        )
                        
                        entry_train_seqimportance = np.concatenate(
                            (
                                meta_entry, 
                                train_loss_seqimportance
                            )
                        )
                        entry_val_seqimportance = np.concatenate(
                            (
                                meta_entry, 
                                val_loss_seqimportance
                            )
                        )
                        
                        df_list_seqimportance.append(
                            pd.DataFrame(
                                {col_name_train: pd.Series(
                                    entry_train_seqimportance
                                )}
                            )
                        )
                        df_list_seqimportance.append(
                            pd.DataFrame(
                                {col_name_val: pd.Series(
                                    entry_val_seqimportance
                                )}
                            )
                        )
                        
                        
                    ### Save budget vs accuracy for AL ### 
                    data = np.rint(100 * np.array(AL_result.budget_usage_hist)).astype(int)
                    sensors = np.rint(100 * np.array(AL_result.sensor_usage_hist)).astype(int)
                    streamtimes = np.rint(100 * np.array(AL_result.streamtime_usage_hist)).astype(int)
                    val_loss = AL_result.val_loss_hist
                    accuracy = np.rint(100 * (1 - np.minimum(1, val_loss / RF_loss))).astype(int)
                    
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
                    
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_data: pd.Series(data)})
                    )
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_sensors: pd.Series(sensors)})
                    )
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_streamtimes: pd.Series(streamtimes)})
                    )
                    budgetvsaccuracy_df_list.append(
                        pd.DataFrame({col_name_accuracy: pd.Series(accuracy)})
                    )
                    
                    picked_times_index_hist = AL_result.picked_times_index_hist
                    picked_spaces_index_hist = AL_result.picked_spaces_index_hist
                    picked_inf_score_hist = AL_result.picked_inf_score_hist
                    
                    
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
                        
                        spacetime_df_list.append(
                            pd.DataFrame({col_name_times: pd.Series(picked_times_list)})
                        )
                        spacetime_df_list.append(
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
                            spacetime_df_list.append(
                                pd.DataFrame({col_name_scores: pd.Series(picked_scores_list)})
                            )

            # create the index column
            df_index = [
                'n_iterations',
                't_iterations',
                'budget_usage',
                'sensor_usage',
                'streamtime_usage',
                'RF_loss',
                'test_loss',
            ]
            
            # concatenate the list of all DataFrames to final (empty) Frame
            result_df = pd.concat(df_list, axis=1)
            
            for i in range(len(result_df) - len(df_index)):
                df_index.append(i)

            # set the index column
            result_df.index = df_index

            # save results to a CSV file
            result_df.to_csv(path_to_results_file)
            
            if HYPER.TEST_SEQUENCE_IMPORTANCE:
                df_index = [
                    'test_loss',
                ]
                
                seqimportance_df = pd.concat(df_list_seqimportance, axis=1)
                
                for i in range(len(seqimportance_df) - len(df_index)):
                    df_index.append(i)
                    
                seqimportance_df.index = df_index
                seqimportance_df.to_csv(path_to_seqimportance_file)
            
            budgetvsaccuracy_df = pd.concat(budgetvsaccuracy_df_list, axis=1)
            budgetvsaccuracy_df.to_csv(path_to_budgetvsaccuracy_file)
            
            spacetime_df = pd.concat(spacetime_df_list, axis=1)
            spacetime_df.to_csv(path_to_spacetime_file)
            

def save_hyper_params(HYPER, raw_data):

    """ Saves all hyper parameter values which are used for calculating these 
    results.
    """

    if HYPER.SAVE_HYPER_PARAMS:

        for pred_index, pred_type in enumerate(HYPER.PRED_LIST_ACT_LRN):
            saving_path = raw_data.path_to_AL_results + pred_type + '/'
            
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)
                
            saving_path += 'hyper.csv'

            # create empty DataFrame
            hyper_df = pd.DataFrame()
            df_list = []
            
            # general parameters
            df_list.append(
                pd.DataFrame({'private_data_access': pd.Series(HYPER.PRIVATE_DATA_ACCESS)})
            )
            df_list.append(
                pd.DataFrame({'test_sequence_importance': pd.Series(
                    HYPER.TEST_SEQUENCE_IMPORTANCE
                )})
            )
            df_list.append(
                pd.DataFrame({'save_act_lrn_results': pd.Series(
                    HYPER.SAVE_ACT_LRN_RESULTS
                )})
            )
            df_list.append(
                pd.DataFrame({'save_hyper_params': pd.Series(
                    HYPER.SAVE_HYPER_PARAMS
                )})
            )
            df_list.append(
                pd.DataFrame({'save_act_lrn_models': pd.Series(
                    HYPER.SAVE_ACT_LRN_MODELS
                )})
            )
            df_list.append(
                pd.DataFrame({'save_act_lrn_test_sample': pd.Series(
                    HYPER.SAVE_ACT_LRN_TEST_SAMPLE
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
                pd.DataFrame({'cluster_ethod_act_lrn': pd.Series(
                    HYPER.CLUSTER_METHOD_ACT_LRN
                )})
            )
            df_list.append(
                pd.DataFrame({'cand_subsample_act_lrn': pd.Series(
                    HYPER.CAND_SUBSAMPLE_ACT_LRN
                )})
            )

            # problem setup parameters
            df_list.append(
                pd.DataFrame({'problem_type': pd.Series(
                    HYPER.PROBLEM_TYPE
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
    AL_result_list, 
    PL_result_list):

    """ Saves a random sample of 1,000 data points from the candidate data pool 
    which were not seen by the actively trained prediction models.
    """

    if HYPER.SAVE_ACT_LRN_TEST_SAMPLE:
        if HYPER.SPATIAL_FEATURES == 'image':
            print('Feature not available')
            return

        for index_pred, pred_type in enumerate(HYPER.PRED_LIST_ACT_LRN):
            saving_path = raw_data.path_to_AL_test_samples + pred_type + '/'
            
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)

            # get method_result_list of currently iterated prediction type
            var_result_list = AL_result_list[index_pred]

            # get random results
            PL_results = PL_result_list[index_pred]
            test_data = PL_results.test_data

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

            for index_var, AL_variable in enumerate(HYPER.QUERY_VARIABLES_ACT_LRN):
                # get variable result list
                method_result_list = var_result_list[index_var]

                for index_method, method in enumerate(
                    HYPER.QUERY_VARIANTS_ACT_LRN
                ):

                    AL_result = method_result_list[index_method]
                    test_data = AL_result.test_data

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
