from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel 
from sklearn.metrics.pairwise import cosine_similarity
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
    
    # Keep a value of False if you don't have access to private data.
    PRIVATE_ACCESS = False

    # Decide whether to test the sequence importance of queried candidates.
    TEST_SEQUENCE_IMPORTANCE = False
    
    # Decide whether or not to save the results and hyper parameters.
    SAVE_ACT_LRN_RESULTS = True
    SAVE_HYPER_PARAMS = True
    SAVE_ACT_LRN_MODELS = True
    SAVE_ACT_LRN_TEST_SAMPLE = True
    
    
    ### 1. Active Learning algorithm ###
    
    # Decide whether to extend initial training data with queried
    # candidates (True) or whether to train on the queried batch
    # only (False) in each iteration of our AL algorithm 
    EXTEND_TRAIN_DATA_ACT_LRN = False

    # Decide whether to remove queried candidates from candidate data pool.
    RED_CAND_DATA_ACT_LRN = True
    
    # Decide whether to delete queried candidates from validation data.
    UPD_VAL_DATA_ACT_LRN = True

    # Decide which prediction types to evaluate. Choose from "spatial",
    # "temporal", "spatio-temporal"
    PRED_LIST_ACT_LRN = [
        #'temporal',
        'spatial',
        #'spatio-temporal'
    ]

    # Decide which methods to evaluate. Choose from "cluster-rnd", 
    # "cluster-far", "cluster-close", "cluster-avg"
    QUERY_VARIANTS_ACT_LRN = [
        'rnd d_c', 
        #'min d_c', 
        #'max d_c', 
        'avg d_c'
    ]

    # Choose AL variables you want to test. Choose from "X_t", "X_s1", "X_st", 
    # "X_(t,s)", "Y_hat_(t,s)", "Y_(t,s)"
    QUERY_VARIABLES_ACT_LRN = [
        #'X_t',
        #'X_s1',
        #'X_st', 
        #'X_(t,s)', 
        'Y_hat_(t,s)', 
        'Y_(t,s)'
    ]

    # Decide how many iterations to go at max for batch AL
    MAX_ITER_ACT_LRN = 10

    # Choose the budget for new training datapoints we can
    # query from the candidate data pool in percentage.
    DATA_BUDGET_ACT_LRN = 0.5

    # Decide which share of the data we take on each iteration,
    # before reusing encoders and newly arranging datapoints.
    CAND_BATCH_SIZE_ACT_LRN = 0.1

    # Decide for how many epochs you want to train your model during AL
    EPOCHS_ACT_LRN = 30

    # Decide how many epochs to have patience on an increasing
    # validation loss when training for early stopping
    PATIENCE_ACT_LRN = 10

    # Decide for which metrics to calculate the problem. Choose
    # from rbf_kernel, laplacian_kernel, cosine_similarity
    METRIC_DISTANCES = [laplacian_kernel]

    # Decide for which clustering mehtods to cluster candidate data
    # points. Choose from KMeans
    METHOD_CLUSTERS = [KMeans]

    # Choose None or a subsample size of uniformly chosen candidates.
    CAND_SUBSAMPLE_ACT_LRN = None
    
    # Decide how many samples to save for each AL test
    SAVED_SAMPLES_ACT_LRN = 1000


    ### 2. Hypothesis and prediciton problem ###

    # Decide how to formulate the problem. Choose from
    # "regression", "classification"
    PROBLEM_TYPE = 'regression'
    
    # Choose from: mean_squared_error, mean_absolute_error,
    #  mean_squared_logarithmic_error, huber, log_cosh
    REGRESSION_LOSS = [tf.keras.losses.mean_squared_error]
    
    # Choose from: sparse_categorical_crossentropy, if
    # PROBLEM_TYPE='classification'.
    CLASSIFICATION_LOSS = [tf.keras.losses.sparse_categorical_crossentropy] 

    # Decide how many classes to consider, if PROBLEM_TYPE='classification'.
    REGRESSION_CLASSES = 200

    # Decide which labels to consider. Choose from "random_scaled",
    # "feature_scaled", "minmax", "original"
    LABELS = 'original'

    # Decide for which years you want to consider electric load profiles.
    # Choose from "2014".
    PROFILE_YEARS = [
        '2014'
    ]

    # Decide which dataset you want to process. You can choose between
    # profiles_100 and profiles_400
    PROFILE_SET = 'profiles_400'

    # Decide how many data points per building-year profile you 
    # want to consider. Choose an integer between 1 and 35040.
    POINTS_PER_PROFILE = 20
    
    # Decide how many time steps to predict consumption 
    # into the future. Resolution is 15 min. 96 ~ 24h.
    PREDICTION_WINDOW = 96

    # Decides on the splitting ratio between
    # training and validation datasets.
    TRAIN_SPLIT = 0.5

    # Decides how many buildings and how much of
    # the time period to separate for testing.
    TEST_SPLIT = 0.5

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

    # set on calling __init__
    INITIALIZATION = None
    LSTM_RECURENT_INITIALIZATION = None

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
    # integer to the power of two or 'None'
    DOWN_SCALE_BUILDING_IMAGES = None

    # decive which meteo data types to consider. Choose from "air_density", 
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
        if not self.PRIVATE_ACCESS:

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

            if self.DOWN_SCALE_BUILDING_IMAGES is not None:

                print(
                    '\n\n You chose DOWN_SCALE_BUILDING_IMAGES={}'.format(
                        self.DOWN_SCALE_BUILDING_IMAGES
                    ),
                    'Note that you run in public access mode and only',
                    'have access to image data that is calculated without',
                    'any down scaling. \n\n'
                )

                self.DOWN_SCALE_BUILDING_IMAGES = None
                
                time.sleep(3)
        
        self.INITIALIZATION = tf.keras.initializers.GlorotNormal(
            seed=random_seed
        )
        self.LSTM_RECURENT_INITIALIZATION = tf.keras.initializers.Orthogonal(
            gain=1.0, 
            seed=random_seed
        )
            
        if self.METRIC_DISTANCES[0] == rbf_kernel:
            self.DISTANCE_METRIC_ACT_LRN = 'Gaussian'
            
        elif self.METRIC_DISTANCES[0] == laplacian_kernel:
            self.DISTANCE_METRIC_ACT_LRN = 'Laplacian'
            
        elif self.METRIC_DISTANCES[0] == cosine_similarity:
            self.DISTANCE_METRIC_ACT_LRN = 'Cosine similarity'

        if self.METHOD_CLUSTERS[0] == KMeans:
            self.CLUSTER_METHOD_ACT_LRN = 'KMeans'
        else:
            print(
                'If you do not own the private data, your experiment will not',
                'succeed in private access mode! Change hyper parameter to',
                'PUBLIC_ACCESS=True'
            )
        

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
