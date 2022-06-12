from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, cosine_similarity
import tensorflow as tf
import time


class HyperParameter:

    """ Keeps hyper parameters together for four categories:
    1. active learning algorithm
    2. hypothesis and prediction model
    3. prediction model
    4. feature engineering
    """
    
    ### General parameters ###
    
    # Keep a value of False if you have access to public data only.
    PRIVATE_DATA_ACCESS = True
    
    # Decide whether and which test to run. Choose from 'main_experiments',
    # 'sequence_importance', 'subsample_importance', 'pointspercluster_importance',
    # 'querybycoordinate_importance'.
    TEST_EXPERIMENT_CHOICE = 'querybycoordinate_importance'
    
    # Decide whether to save results, hyper paramters, models and sample data.
    SAVE_RESULTS = True
    
    
    ### 1. Active Learning algorithm ###
    
    # Decide whether to extend initial training data with queried
    # candidates (True) or whether to train on the queried batch
    # only (False) in each iteration of our AL algorithm 
    EXTEND_TRAIN_DATA_ACT_LRN = False

    # Decide whether to remove queried candidates from candidate data pool.
    RED_CAND_DATA_ACT_LRN = True
    
    # Decide whether to delete queried candidates from validation data.
    UPD_VAL_DATA_ACT_LRN = True

    # Decide which prediction types to evaluate. Choose from 'spatial',
    # 'temporal' and 'spatio-temporal'
    PRED_TYPE_ACT_LRN = 'spatio-temporal'
    
    # Choose AL variables you want to test. Choose from 'X_t', 'X_s1', 'X_st', 
    # 'X_(t,s)', 'Y_hat_(t,s)', 'Y_(t,s)'
    QUERY_VARIABLES_ACT_LRN = [
        'X_st', 
        #'X_(t,s)', 
        #'Y_hat_(t,s)', 
        'Y_(t,s)'
    ]
    
    # Decide which active learning variants to evaluate. Choose from 'rnd d_c', 
    # 'min d_c', 'max d_c' and 'avg_dc'.
    QUERY_VARIANTS_ACT_LRN = [
        #'rnd d_c', 
        #'min d_c', 
        'max d_c', 
        'avg d_c'
    ]
    
    # Decide how many iterations to go at max for batch active learning
    N_ITER_ACT_LRN = 10
    
    # Choose which share of candidate data pool we want to set as our entire
    # data budget for new queries form it
    DATA_BUDGET_ACT_LRN = 0.5
    
    # Heuristics. Choose a value between 0 and 1. A value of 0 creates one cluster
    # only for querying candidates. A value of 1 creates one cluster for each point
    POINTS_PER_CLUSTER_ACT_LRN = 1
    
    # Heuristics. Choose a value between 0 and 1. A value of 0 creates a candidate
    # subsample that is equal to 
    CAND_SUBSAMPLE_ACT_LRN = 1
    
    # Decide how many epochs you want to train your model during active learning
    EPOCHS_ACT_LRN = 30
    
    # Decide how many epochs to have patience on an increasing
    # validation loss when training for early stopping
    PATIENCE_ACT_LRN = 10
    
    # Decide for which metrics to calculate the problem. Choose
    # from 'Gaussian', 'Laplacian' and 'CosineSimilarity'
    DISTANCE_METRIC_ACT_LRN = 'Laplacian'
    
    # Decide for which clustering mehtods to cluster candidate data
    # points. Choose from 'KMeans', 'MiniBatchKMeans'
    CLUSTER_METHOD_ACT_LRN = 'KMeans'
    
    # Decide how many samples to save for each AL test
    SAVED_SAMPLES_ACT_LRN = 1000
    
    
    ### 2. Hypothesis and prediciton problem ###

    # Decide how to formulate the problem. Choose from
    # 'regression' and 'classification'
    PROBLEM_TYPE = 'regression'
    
    # Choose from: 'mean_squared_error', 'mean_absolute_error',
    # 'mean_squared_logarithmic_error', 'huber', 'log_cosh'
    REGRESSION_LOSS_NAME = 'mean_squared_error'
    
    # Decide how many classes to consider. Only applies if 
    # PROBLEM_TYPE='classification'.
    REGRESSION_CLASSES = 200

    # Decide which labels to consider. Choose from 'random_scaled',
    # 'feature_scaled', 'minmax', 'original'
    LABELS = 'original'

    # Decide for which years you want to consider electric load profiles.
    # Choose from '2014'.
    PROFILE_YEARS = [
        '2014'
    ]
    
    # Decide which dataset you want to process. You can choose between
    # 'profiles_100' and 'profiles_400'
    PROFILE_SET = 'profiles_100'

    # Decide how many building-year profiles you want to
    # consider for each year. Choose a share between 0 and 1. A value of 1
    # corresponds to about 100 profiles for the profiles_100 and 400 profiles
    # for the profiles_400 dataset
    PROFILES_PER_YEAR = 1
    
    # Decide how many data points per building-year profile you 
    # want to consider. Choose a share between 0 and 1. A value of 0.001 
    # corresponds to approximately 35 points per profile
    POINTS_PER_PROFILE = 0.0005
    
    # Decide how many time steps to predict consumption into the future.
    # Resolution is 15 min. A values of 96 corresponds to 24h.
    PREDICTION_WINDOW = 96

    # Decides on the splitting ratio between training and validation datasets.
    TRAIN_SPLIT = 0.3

    # Decides how many buildings and how much of the time period to separate for 
    # testing.
    TEST_SPLIT = 0.7

    # Decide in which frequency to do train-validation split. 1 equals 
    # one datapoint per bin, 0.5 equals two datapoints per bin.
    SPLIT_INTERAVALS = 0.05
    
    
    ### 3. Prediction model ###

    # Decide for how many epochs you want to train your model.
    EPOCHS = 30

    # Decide how many epochs to have patience on not increasing
    # validation loss during training before early stopping.
    PATIENCE = 10

    # Decide how large your data batch size should be during training. 
    # Choose something to the power of 2.
    BATCH_SIZE = 16

    # Decide how many neural network layers you want to use
    # for encoding features.
    ENCODER_LAYERS = 1

    # Decide on the dimension of the encoding vectors.
    ENCODING_NODES_X_t = 100
    ENCODING_NODES_X_s = 100
    ENCODING_NODES_X_st = 100
    ENCODING_NODES_X_joint = 100
    
    # Decide which activation function to use on last encoding layer.
    # Choose from None, "relu", "tanh", "selu", "elu", "exponential".
    ENCODING_ACTIVATION = 'relu'
    
    # Decide how many layers you want to use after encoders.
    # This is your network depth.
    NETWORK_LAYERS = 1
    
    # Decide how many nodes per layer you want to use.
    NODES_PER_LAYER_DENSE = 1000
    FILTERS_PER_LAYER_CNN = 16
    STATES_PER_LAYER_LSTM = 200
    
    # Decide which layers to use for X_st inputs. Choose one
    # from "ANN", "CNN", "LSTM".
    LAYER_TYPE_X_ST = 'CNN'

    # Decide which activation function to use in each layer. Choose
    # from None, "relu", "tanh", "selu", "elu", "exponential".
    DENSE_ACTIVATION = 'relu'
    CNN_ACTIVATION = 'relu'
    LSTM_ACTIVATION = 'tanh'

    # Decide how to initiliaze weights for Conv1D, Conv2D, LSTM and
    # Dense layers. Choose from "glorot_uniform", "glorot_normal", 
    # "random_normal", "random_uniform", "truncated_normal"
    INITIALIZATION_METHOD = 'glorot_normal'
    INITIALIZATION_METHOD_LSTM = 'orthogonal'

    # decide whether or not to use batch normalization on each layer in your NN. 
    # Choose between True or False
    BATCH_NORMALIZATION = False

    # decide how to regularize weights. Choose from None, 'l1', 'l2', 'l1_l2'
    REGULARIZER = 'l1_l2'
    
    
    ### 4. Feature engineering ###

    # decide which time stamp information to consider. Choose from: '15min',
    # 'hour', 'day', 'month', 'year'
    TIMESTAMP_DATA = [
        '15min', 
        'hour', 
        'day', 
        'month'
    ]

    # decide how to encode time stamp data. Choose one of 'ORD', 'ORD-1D'
    # or 'OHE'
    TIME_ENCODING = 'ORD'

    # decide how to treat aerial imagery. Choose one from 'average',
    # 'histogram', 'image'
    SPATIAL_FEATURES = 'histogram'

    # set the number of histogram bins that you want to use. Applied if 
    # SPATIAL_FEATURES = 'histogram'
    HISTO_BINS = 100

    # decide whether you want to consider underlying RGB images in grey-scale
    GREY_SCALE = False

    # decide whether and how to downscale spatial imagery data. Choose any 
    # integer to the power of two or 0 for no downscaling
    DOWN_SCALE_BUILDING_IMAGES = 0

    # Decide which meteo data types to consider. Choose from "air_density", 
    # "cloud_cover",  "precipitation", "radiation_surface", "radiation_toa", 
    # "snow_mass", "snowfall", "temperature", "wind_speed"
    METEO_TYPES = [
        'air_density',
        'cloud_cover',
        'precipitation',
        'radiation_surface',
        'radiation_toa',
        'snow_mass',
        'snowfall',
        'temperature',
        'wind_speed',
    ]

    # choose past time window for the meteo data. Resolution is hourly.
    HISTORY_WINDOW_METEO = 24

    # decide whether or not to normalize features
    NORMALIZATION = True

    # decide whether to standardize features to zero mean and unit variance.
    STANDARDIZATION = True


    ### Methods ###

    def __init__(self, random_seed):

        """ Sets the weight initializer for tensorflow layers. """

        # check if chosen hyper parameters are valid and correct if necessary
        if not self.PRIVATE_DATA_ACCESS:

            if self.LABELS == 'original':

                print(
                    '\n You run in public access mode, and only have access', 
                    'to public data. HYPER.LABELS is set to "feature_scaled".'
                )

                self.LABELS = 'feature_scaled'

            if self.SPATIAL_FEATURES == 'image':

                def ask_for_spatial_features_input(self):

                    response = input(
                        '\n You run in public access mode, and only '
                        + 'have access to image histograms. Please enter '
                        + 'h to set SPATIAL_FEATURES="histogram" or '
                        + 'a for SPATIAL_FEATURES="average":'
                    )
                    
                    if response == 'h':
                        self.SPATIAL_FEATURES = 'histogram'
                        return True
                        
                    elif response == 'a':
                        self.SPATIAL_FEATURES = 'average'
                        return True
                        
                    else:
                        print('\n Input not recognized. Please try again.')
                        return False
                 
                a = 0
                
                while a < 1:
                
                    response = ask_for_spatial_features_input(self)
                    
                    if response:
                    
                        break

            if self.HISTO_BINS != 100:

                print(
                    '\n\n You chose HISTO_BINS={}. Note that'.format(
                        self.HISTO_BINS
                    ),
                    'you run in public access mode and only have access to', 
                    'image histograms that are',
                    'calculated with HISTO_BINS=100. \n\n'
                )

                self.HISTO_BINS = 100

                time.sleep(3)

            if self.DOWN_SCALE_BUILDING_IMAGES != 0:

                print(
                    '\n\n You chose DOWN_SCALE_BUILDING_IMAGES={}'.format(
                        self.DOWN_SCALE_BUILDING_IMAGES
                    ),
                    'Note that you run in public access mode and only',
                    'have access to image data that is calculated without',
                    'any down scaling. \n\n'
                )

                self.DOWN_SCALE_BUILDING_IMAGES = 0
                
                time.sleep(3)
        
        else:
            print(
                'If you do not own the private data, your experiment will not',
                'succeed in private access mode! Change hyper parameter to',
                'PUBLIC_ACCESS=True'
            )
        
        # Set the initialization for regular weights.
        if self.INITIALIZATION_METHOD == 'glorot_normal':
            self.INITIALIZATION = tf.keras.initializers.GlorotNormal(
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD == 'glorot_uniform':
            self.INITIALIZATION = tf.keras.initializers.GlorotUniform(
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD == 'random_uniform':
            self.INITIALIZATION = tf.keras.initializers.RandomUniform(
                minval=-0.05, 
                maxval=0.05, 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD == 'truncated_normal':
            self.INITIALIZATION = tf.keras.initializers.TruncatedNormal(
                mean=0.0, 
                stddev=0.05, 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD == 'orthogonal':
            self.INITIALIZATION = tf.keras.initializers.Orthogonal(
                gain=1.0, 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD == 'variance_scaling':
            self.INITIALIZATION = tf.keras.initializers.VarianceScaling(
                scale=1.0, 
                mode='fan_in', 
                distribution='truncated_normal', 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD == 'random_normal':
            self.INITIALIZATION = tf.keras.initializers.RandomNormal(
                mean=0.0, 
                stddev=0.05, 
                seed=random_seed
            )
        
        
        # Set the initialization for LSTM layer weights.
        if self.INITIALIZATION_METHOD_LSTM == 'glorot_normal':
            self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.GlorotNormal(
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD_LSTM == 'glorot_uniform':
            self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.GlorotUniform(
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD_LSTM == 'random_uniform':
            self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.RandomUniform(
                minval=-0.05, 
                maxval=0.05, 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD_LSTM == 'truncated_normal':
            self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.TruncatedNormal(
                mean=0.0, 
                stddev=0.05, 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD_LSTM == 'orthogonal':
            self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.Orthogonal(
                gain=1.0, 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD_LSTM == 'variance_scaling':
            self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.VarianceScaling(
                scale=1.0, 
                mode='fan_in', 
                distribution='truncated_normal', 
                seed=random_seed
            )
        elif self.INITIALIZATION_METHOD_LSTM == 'random_normal':
            self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.RandomNormal(
                mean=0.0, 
                stddev=0.05, 
                seed=random_seed
            )
        
        # Set distance metric
        if self.DISTANCE_METRIC_ACT_LRN == 'Gaussian':
            self.METRIC_DISTANCES = [rbf_kernel]
        elif self.DISTANCE_METRIC_ACT_LRN == 'Laplacian':
            self.METRIC_DISTANCES = [laplacian_kernel]
        elif self.DISTANCE_METRIC_ACT_LRN == 'CosineSimilarity':
            self.METRIC_DISTANCES = [cosine_similarity]
        
        # set cluster method
        if self.CLUSTER_METHOD_ACT_LRN == 'KMeans':
            self.METHOD_CLUSTERS = [KMeans]
        elif self.CLUSTER_METHOD_ACT_LRN == 'MiniBatchKMeans':
            self.METHOD_CLUSTERS = [MiniBatchKMeans]
        
        # Set classification loss. Only applies if PROBLEM_TYPE='classification'.
        self.CLASSIFICATION_LOSS = [tf.keras.losses.sparse_categorical_crossentropy] 
        
        # Set regression loss function
        if self.REGRESSION_LOSS_NAME == 'mean_squared_error':
            self.REGRESSION_LOSS = [tf.keras.losses.mean_squared_error]
        elif self.REGRESSION_LOSS_NAME == 'mean_absolute_error':
            self.REGRESSION_LOSS = [tf.keras.losses.mean_absolute_error]
        elif self.REGRESSION_LOSS_NAME == 'mean_squared_logarithmic_error':
            self.REGRESSION_LOSS = [tf.keras.losses.mean_squared_logarithmic_error]
        elif self.REGRESSION_LOSS_NAME == 'huber':
            self.REGRESSION_LOSS = [tf.keras.losses.huber]
        elif self.REGRESSION_LOSS_NAME == 'log_cosh':
            self.REGRESSION_LOSS = [tf.keras.losses.log_cosh] 
        
        # set heuristic importance test value lists
        if (
            self.TEST_EXPERIMENT_CHOICE != 'main_experiments' and 
            self.TEST_EXPERIMENT_CHOICE != 'sequence_importance'
        ):
            self.CAND_SUBSAMPLE_TEST_LIST = [
                0.3,
                0.5,
                0.7,
                1
            ]
            self.POINTS_PERCLUSTER_TEST_LIST = [
                0,
                0.3,
                0.5,
                1
            ]
        
        if self.TEST_EXPERIMENT_CHOICE == 'querybycoordinate_importance':
            print(
                'Caution!! You decided to test query by coordinates through '
                'setting TEST_QUERYBYCOORDINATE_IMPORTANCE=True. This will '
                'set the following hyper paramters before performing ADL: \n',
                'QUERY_VARIABLES_ACT_LRN = ["X_t", "X_s1"] \n',
                'TEST_HEURISTIC_IMPORTANCE = ["max d_c"] \n',
                'POINTS_PER_CLUSTER_ACT_LRN = 0 \n',
            )
            
            self.QUERY_VARIABLES_ACT_LRN = ['X_t', 'X_s1']
            self.QUERY_VARIANTS_ACT_LRN = ['max d_c']
            self.POINTS_PER_CLUSTER_ACT_LRN = 0
            
            
    def set_act_lrn_params(self):

        """ Should be called only after training initial model. Resets the 
        general neural network training properties of patience and epochs to 
        those that we chose to be used during active learning.
        """

        self.PATIENCE = self.PATIENCE_ACT_LRN
        self.EPOCHS = self.EPOCHS_ACT_LRN

    def show_attributes(self):

        """ Prints out the key-value pairs of all attributes of this class. """

        for attr, value in self.__dict__.items():
            print(attr, ':', value)
