import math
import timeit
import random

import numpy as np
import tensorflow as tf
import scipy

from sklearn.preprocessing import OrdinalEncoder
from data import Dataset
from prediction import train_model, test_model
from prediction import load_encoder_and_predictor_weights
from prediction import initialize_optimizer
import saveresults


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
    n_clusters = math.ceil(
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
    results = saveresults.ActLrnResults(
        train_hist,
        val_hist,
        test_loss,
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
