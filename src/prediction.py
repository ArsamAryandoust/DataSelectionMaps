import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor


class EncodersAndPredictor:

    """ Keeps prediction and encoding models together. """

    def __init__(
        self, 
        X_t_encoder, 
        X_s1_encoder, 
        X_st_encoder, 
        X_joint_encoder, 
        prediction_model
    ):

        self.X_t_encoder = X_t_encoder
        self.X_s1_encoder = X_s1_encoder
        self.X_st_encoder = X_st_encoder
        self.X_joint_encoder = X_joint_encoder
        self.prediction_model = prediction_model


def initialize_optimizer(HYPER):
    
    """ Sets the loss objective, epending on whether we solve the prediction 
    task as a regression or a classification problem. Note: the distinction 
    between 'loss_function' and 'loss_object' is because of classification loss. 
    We still want to evaluate the predictions with a regression loss if we 
    transform our regression to a classification problem and use the respective 
    cross entropy categorical loss.
    """
    
    # choose a loss function
    if HYPER.PROBLEM_TYPE == 'regression':
        loss_object = HYPER.REGRESSION_LOSS[0]
        
    elif HYPER.PROBLEM_TYPE == 'classification':
        loss_object = HYPER.CLASSIFICATION_LOSS[0]
        
    # set an optimization algorithm
    optimizer = tf.keras.optimizers.RMSprop()
    
    # set the loss metric for training and testing. This is always for regression.
    loss_function = HYPER.REGRESSION_LOSS[0]
    
    # set that we want to calculate the mean with respect to individual losses
    mean_loss = tf.keras.metrics.Mean(name='mean_loss_train_test')
        
    return loss_object, optimizer, loss_function, mean_loss


def create_and_train_RF(HYPER, train_data):

    """ Creates, trains and returns a Random Forest regression model. """

    # create a random forest regression model
    RF_regr = RandomForestRegressor(
        max_depth=None, n_estimators=200, oob_score=True, n_jobs=-1
    )

    # train the random forest regression model
    if HYPER.SPATIAL_FEATURES == 'image':
        RF_regr.fit(
            np.concatenate(
                (
                    train_data.X_t,
                    train_data.X_s[:, 1:],
                    np.reshape(
                        train_data.X_st,
                        (
                            train_data.X_st.shape[0],
                            train_data.X_st.shape[1] * train_data.X_st.shape[2],
                        ),
                        order='F',
                    ),
                ),
                axis=1,
            ),
            train_data.Y,
        )
        
    else:
        RF_regr.fit(
            np.concatenate(
                (
                    train_data.X_t,
                    np.reshape(
                        train_data.X_s1,
                        (
                            train_data.X_s1.shape[0],
                            train_data.X_s1.shape[1] * train_data.X_s1.shape[2],
                        ),
                    ),
                    np.reshape(
                        train_data.X_st,
                        (
                            train_data.X_st.shape[0],
                            train_data.X_st.shape[1] * train_data.X_st.shape[2],
                        ),
                        order='F',
                    ),
                ),
                axis=1,
            ),
            train_data.Y,
        )

    return RF_regr


def predict_with_RF(HYPER, RF_regr, dataset):

    """ Returns predictions on dataset, givne RF_regr model. """

    if HYPER.SPATIAL_FEATURES == 'image':
        predictions = RF_regr.predict(
            np.concatenate(
                (
                    dataset.X_t,
                    dataset.X_s[:, 1:],
                    np.reshape(
                        dataset.X_st,
                        (
                            dataset.X_st.shape[0],
                            dataset.X_st.shape[1] * dataset.X_st.shape[2],
                        ),
                        order='F',
                    ),
                ),
                axis=1,
            )
        )
        
    else:
        predictions = RF_regr.predict(
            np.concatenate(
                (
                    dataset.X_t,
                    np.reshape(
                        dataset.X_s1,
                        (
                            dataset.X_s1.shape[0],
                            dataset.X_s1.shape[1] * dataset.X_s1.shape[2],
                        ),
                    ),
                    np.reshape(
                        dataset.X_st,
                        (
                            dataset.X_st.shape[0],
                            dataset.X_st.shape[1] * dataset.X_st.shape[2],
                        ),
                        order='F',
                    ),
                ),
                axis=1,
            )
        )

    return predictions


def save_prediction_model(HYPER, raw_data, model, model_name):

    """ Saves the passed prediction model when called. """
    
    if HYPER.SAVE_ACT_LRN_MODELS:
    
        for pred_type in HYPER.PRED_LIST_ACT_LRN:
            saving_path = raw_data.path_to_AL_models + pred_type + '/'
            
            if not os.path.exists(saving_path):
                os.mkdir(saving_path)
        
            saving_path += model_name + '.h5'
            model.save(saving_path)
            
            
def save_encoder_and_predictor_weights(HYPER, raw_data, models):

    """ Saves the encoder and the prediction model weights of the 
    EncodersAndPredictor object that is passed, in a path that is contained in 
    the RawData object under attribute path_to_encoder_weights.
    """
    
    for pred_type in HYPER.PRED_LIST_ACT_LRN:
        saving_path = raw_data.path_to_encoder_weights + pred_type + '/'
        
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)
    
        # iterate simultaneously over models and their names
        for model_name, tf_model in models.__dict__.items():

            # create the full path for saving currently iterated model
            path_to_model_weights = saving_path + model_name

            # save currently iterated model
            tf_model.save_weights(path_to_model_weights)
            
def load_encoder_and_predictor_weights(raw_data, models, pred_type):
    
    """ Loads the weights for all passed models. """

    # iterate simultaneously over models and their names
    for model_name, tf_model in models.__dict__.items():

        # create the full path for saving currently iterated model
        loading_path = (
            raw_data.path_to_encoder_weights 
            + pred_type 
            + '/' 
            + model_name
        )

        # load currently iterated model
        exec(
            "models.{}.load_weights(loading_path)".format(
                model_name
            )
        )
        
    return models
    

def plot_true_vs_prediction(figtitle, test_data_Y, predictions):

    """ Visualizes predictions vs. true values. """

    plot_rows = 3
    plot_clos = 3

    # create a matplotlib.pyplot.subplots figure
    fig, ax = plt.subplots(plot_rows, plot_clos, figsize=(16, 16))

    # set the figtitle
    fig.suptitle(figtitle, fontsize=16)

    # pick at random a set of integers for visualization
    data_indices = np.random.randint(
        0, 
        len(test_data_Y), 
        plot_rows * plot_clos
    )

    # create a variable for iteratively adding number of subplots
    subplot_counter = 0

    # iterate over number of rows
    for i in range(plot_rows):

        # iterate over number of columns
        for j in range(plot_clos):

            # choose currently iterated random index
            data_index = data_indices[subplot_counter]

            # plot the true values
            plot1 = ax[i, j].plot(test_data_Y[data_index])

            # plot the predicted values
            plot2 = ax[i, j].plot(predictions[data_index])

            # increment the subplot_counter
            subplot_counter += 1

    # add a figure legend
    fig.legend(
        [plot1, plot2],
        labels=['true load profile', 'predicted load profile'],
        fontsize=16,
    )


def plot_RF_feature_importance(
    HYPER, 
    raw_data, 
    RF_feature_importance, 
    training_data
):

    """ Plots feature importance scores. """

    # create an empty feature name list
    feature_name_list = []
    feature_score_list = []

    # add temporal features
    feature_name_list.append('temporal')
    if HYPER.TIME_ENCODING == 'ORD':
        feature_score_list.append(
            np.average(
                RF_feature_importance[: len(HYPER.TIMESTAMP_DATA)]
            ).tolist()
        )
        for name, score in zip(
            HYPER.TIMESTAMP_DATA,
            RF_feature_importance[: len(HYPER.TIMESTAMP_DATA)].tolist(),
        ):
            feature_name_list.append(name)
            feature_score_list.append(score)
        RF_feature_importance = RF_feature_importance[
            len(HYPER.TIMESTAMP_DATA):
        ]
    elif HYPER.TIME_ENCODING == 'ORD-1D':
        feature_score_list.append(RF_feature_importance[0].tolist())
        RF_feature_importance = RF_feature_importance[1:]
    else:
        print(
            'Time encoding is OHE, and not implemented for',
             'calculating feature importance'
        )

    # add spatial features
    feature_name_list.append('spatial')
    if HYPER.SPATIAL_FEATURES == 'image':
        feature_score_list.append(RF_feature_importance[0].tolist())
        RF_feature_importance = RF_feature_importance[1:]
        
    elif HYPER.SPATIAL_FEATURES == 'average':
        feature_score_list.append(
            np.average(
                RF_feature_importance[:raw_data.n_channels]
                )
        )
        RF_feature_importance = RF_feature_importance[raw_data.n_channels:]
        
    elif HYPER.SPATIAL_FEATURES == 'histogram':
        feature_score_list.append(
            np.average(
                RF_feature_importance[: HYPER.HISTO_BINS]
            )
        )
        RF_feature_importance = RF_feature_importance[HYPER.HISTO_BINS :]

    # add spatio-temporal features
    feature_name_list.append('spatio-temporal')
    feature_score_list.append(np.average(RF_feature_importance))
    
    for name in HYPER.METEO_TYPES:
        feature_name_list.append(name)
        feature_score_list.append(
            np.average(RF_feature_importance[: HYPER.HISTORY_WINDOW_METEO])
        )
        RF_feature_importance = RF_feature_importance[
            HYPER.HISTORY_WINDOW_METEO:
        ]

    n_features = len(feature_name_list)
    plt.figure(figsize=(16, n_features * 0.5))
    plt.title('Feature importance Random Forest')
    plt.barh(
        range(n_features), 
        feature_score_list, 
        tick_label=feature_name_list
    )


def build_prediction_model(
    HYPER, 
    raw_data, 
    train_data, 
    silent=False, 
    plot=True
):

    """ Builds encoders and prediction model according to multiple hyper 
    parameters. Returns models and encoders bundled as an EncodersAndPredictor 
    object.
    """

    if not silent:
        # tell us what we do
        print('Building prediction model')

    X_t_example = train_data.X_t[0]
    X_st_example = train_data.X_st[0]
    Y_example = train_data.Y[0]
    
    if HYPER.SPATIAL_FEATURES == 'image':
        X_s1_example = raw_data.building_imagery_data_list[0]
    else:
        X_s1_example = train_data.X_s1[0]


    ### Create the input layers ###
    
    X_t_input = tf.keras.Input(shape=X_t_example.shape, name='X_t')
    X_s1_input = tf.keras.Input(shape=X_s1_example.shape, name='X_s1')
    X_st_input = tf.keras.Input(shape=X_st_example.shape, name='X_st')


    ### Create the hidden layers ###

    ### Encode X_t ###
    
    if HYPER.ENCODER_LAYERS == 0:
        X_t = tf.keras.layers.Flatten()(X_t_input)

    else:
        X_t = tf.keras.layers.Dense(
            HYPER.NODES_PER_LAYER_DENSE,
            activation=HYPER.DENSE_ACTIVATION,
            kernel_initializer=HYPER.INITIALIZATION,
            kernel_regularizer=HYPER.REGULARIZER,
        )(X_t_input)
        
        if HYPER.BATCH_NORMALIZATION:
            X_t = tf.keras.layers.BatchNormalization()(X_t)
            
        for i in range(HYPER.ENCODER_LAYERS - 1):
            X_t = tf.keras.layers.Dense(
                HYPER.NODES_PER_LAYER_DENSE,
                activation=HYPER.DENSE_ACTIVATION,
                kernel_initializer=HYPER.INITIALIZATION,
                kernel_regularizer=HYPER.REGULARIZER,
            )(X_t)
            
            if HYPER.BATCH_NORMALIZATION:
                X_t = tf.keras.layers.BatchNormalization()(X_t)
                
        X_t = tf.keras.layers.Flatten()(X_t)

    X_t = tf.keras.layers.Dense(
        HYPER.ENCODING_NODES_X_t,
        activation=HYPER.ENCODING_ACTIVATION,
        kernel_initializer=HYPER.INITIALIZATION,
        kernel_regularizer=HYPER.REGULARIZER,
    )(X_t)
    
    if HYPER.BATCH_NORMALIZATION:
        X_t = tf.keras.layers.BatchNormalization()(X_t)


    ### Encode X_s1 and X_s2 ###
    
    if HYPER.SPATIAL_FEATURES == 'image':
        if HYPER.ENCODER_LAYERS == 0:
            X_s1 = tf.keras.layers.Conv2D(
                HYPER.FILTERS_PER_LAYER_CNN,
                (2, 2),
                activation=HYPER.CNN_ACTIVATION,
                kernel_initializer=HYPER.INITIALIZATION,
                kernel_regularizer=HYPER.REGULARIZER,
            )(X_s1_input)
            
            if HYPER.BATCH_NORMALIZATION:
                X_s1 = tf.keras.layers.BatchNormalization()(X_s1)
                
        else:
            X_s1 = tf.keras.layers.Conv2D(
                HYPER.FILTERS_PER_LAYER_CNN,
                (2, 2),
                activation=HYPER.CNN_ACTIVATION,
                kernel_initializer=HYPER.INITIALIZATION,
                kernel_regularizer=HYPER.REGULARIZER,
            )(X_s1_input)
            
            if HYPER.BATCH_NORMALIZATION:
                X_s1 = tf.keras.layers.BatchNormalization()(X_s1)
                
            X_s1 = tf.keras.layers.MaxPooling2D((2, 2))(X_s1)
            
            for i in range(HYPER.ENCODER_LAYERS - 1):
                X_s1 = tf.keras.layers.Conv2D(
                    HYPER.FILTERS_PER_LAYER_CNN,
                    (2, 2),
                    activation=HYPER.CNN_ACTIVATION,
                    kernel_initializer=HYPER.INITIALIZATION,
                    kernel_regularizer=HYPER.REGULARIZER,
                )(X_s1)
                
                if HYPER.BATCH_NORMALIZATION:
                    X_s1 = tf.keras.layers.BatchNormalization()(X_s1)
                    
                X_s1 = tf.keras.layers.MaxPooling2D((2, 2))(X_s1)

        X_s1 = tf.keras.layers.Flatten()(X_s1)
        X_s1 = tf.keras.layers.Dense(
            HYPER.ENCODING_NODES_X_s,
            activation=HYPER.ENCODING_ACTIVATION,
            kernel_initializer=HYPER.INITIALIZATION,
            kernel_regularizer=HYPER.REGULARIZER,
        )(X_s1)
        
        if HYPER.BATCH_NORMALIZATION:
            X_s1 = tf.keras.layers.BatchNormalization()(X_s1)

    else:

        if HYPER.ENCODER_LAYERS == 0:
            X_s1 = tf.keras.layers.Flatten()(X_s1_input)

        else:
            X_s1 = tf.keras.layers.Dense(
                HYPER.NODES_PER_LAYER_DENSE,
                activation=HYPER.DENSE_ACTIVATION,
                kernel_initializer=HYPER.INITIALIZATION,
                kernel_regularizer=HYPER.REGULARIZER,
            )(X_s1_input)
            
            if HYPER.BATCH_NORMALIZATION:
                X_s1 = tf.keras.layers.BatchNormalization()(X_s1)
                
            for i in range(HYPER.ENCODER_LAYERS - 1):
                X_s1 = tf.keras.layers.Dense(
                    HYPER.NODES_PER_LAYER_DENSE,
                    activation=HYPER.DENSE_ACTIVATION,
                    kernel_initializer=HYPER.INITIALIZATION,
                    kernel_regularizer=HYPER.REGULARIZER,
                )(X_s1)
                
                if HYPER.BATCH_NORMALIZATION:
                    X_s1 = tf.keras.layers.BatchNormalization()(X_s1)
                    
            X_s1 = tf.keras.layers.Flatten()(X_s1)

        X_s1 = tf.keras.layers.Dense(
            HYPER.ENCODING_NODES_X_s,
            activation=HYPER.ENCODING_ACTIVATION,
            kernel_initializer=HYPER.INITIALIZATION,
            kernel_regularizer=HYPER.REGULARIZER,
        )(X_s1)
        
        if HYPER.BATCH_NORMALIZATION:
            X_s1 = tf.keras.layers.BatchNormalization()(X_s1)


    ### Encode X_st ###
    
    if HYPER.ENCODER_LAYERS == 0:
        X_st = tf.keras.layers.Flatten()(X_st_input)

    else:
        if HYPER.LAYER_TYPE_X_ST == 'ANN':
            X_st = tf.keras.layers.Dense(
                HYPER.NODES_PER_LAYER_DENSE,
                activation=HYPER.DENSE_ACTIVATION,
                kernel_initializer=HYPER.INITIALIZATION,
                kernel_regularizer=HYPER.REGULARIZER,
            )(X_st_input)
            
            if HYPER.BATCH_NORMALIZATION:
                X_st = tf.keras.layers.BatchNormalization()(X_st)
                
            for i in range(HYPER.ENCODER_LAYERS - 1):
                X_st = tf.keras.layers.Dense(
                    HYPER.NODES_PER_LAYER_DENSE,
                    activation=HYPER.DENSE_ACTIVATION,
                    kernel_initializer=HYPER.INITIALIZATION,
                    kernel_regularizer=HYPER.REGULARIZER,
                )(X_st)
                
                if HYPER.BATCH_NORMALIZATION:
                    X_st = tf.keras.layers.BatchNormalization()(X_st)

        elif HYPER.LAYER_TYPE_X_ST == 'CNN':
            X_st = tf.keras.layers.Conv1D(
                HYPER.FILTERS_PER_LAYER_CNN,
                2,
                activation=HYPER.CNN_ACTIVATION,
                kernel_initializer=HYPER.INITIALIZATION,
                kernel_regularizer=HYPER.REGULARIZER,
            )(X_st_input)
            
            if HYPER.BATCH_NORMALIZATION:
                X_st = tf.keras.layers.BatchNormalization()(X_st)
                
            for i in range(HYPER.ENCODER_LAYERS - 1):
                X_st = tf.keras.layers.Conv1D(
                    HYPER.FILTERS_PER_LAYER_CNN,
                    2,
                    activation=HYPER.CNN_ACTIVATION,
                    kernel_initializer=HYPER.INITIALIZATION,
                    kernel_regularizer=HYPER.REGULARIZER,
                )(X_st)
                
                if HYPER.BATCH_NORMALIZATION:
                    X_st = tf.keras.layers.BatchNormalization()(X_st)

        elif HYPER.LAYER_TYPE_X_ST == 'LSTM':
            if HYPER.ENCODER_LAYERS == 1:
                X_st = tf.keras.layers.LSTM(
                    HYPER.STATES_PER_LAYER_LSTM,
                    activation=HYPER.LSTM_ACTIVATION,
                    kernel_initializer=HYPER.INITIALIZATION,
                    kernel_regularizer=HYPER.REGULARIZER,
                )(X_st_input)
                
                if HYPER.BATCH_NORMALIZATION:
                    X_st = tf.keras.layers.BatchNormalization()(X_st)
                    
            else:
                X_st = tf.keras.layers.LSTM(
                    HYPER.STATES_PER_LAYER_LSTM,
                    return_sequences=True,
                    activation=HYPER.LSTM_ACTIVATION,
                    kernel_initializer=HYPER.INITIALIZATION,
                    kernel_regularizer=HYPER.REGULARIZER,
                )(X_st_input)
                
                if HYPER.BATCH_NORMALIZATION:
                    X_st = tf.keras.layers.BatchNormalization()(X_st)
                    
                for i in range(HYPER.ENCODER_LAYERS - 2):
                    X_st = tf.keras.layers.LSTM(
                        HYPER.STATES_PER_LAYER_LSTM,
                        return_sequences=True,
                        activation=HYPER.LSTM_ACTIVATION,
                        kernel_initializer=HYPER.INITIALIZATION,
                        kernel_regularizer=HYPER.REGULARIZER,
                    )(X_st)
                    
                    if HYPER.BATCH_NORMALIZATION:
                        X_st = tf.keras.layers.BatchNormalization()(X_st)
                        
                X_st = tf.keras.layers.LSTM(
                    HYPER.STATES_PER_LAYER_LSTM,
                    activation=HYPER.LSTM_ACTIVATION,
                    kernel_initializer=HYPER.INITIALIZATION,
                    kernel_regularizer=HYPER.REGULARIZER,
                )(X_st)
                
                if HYPER.BATCH_NORMALIZATION:
                    X_st = tf.keras.layers.BatchNormalization()(X_st)

        X_st = tf.keras.layers.Flatten()(X_st)

    X_st = tf.keras.layers.Dense(
        HYPER.ENCODING_NODES_X_st,
        activation=HYPER.ENCODING_ACTIVATION,
        kernel_initializer=HYPER.INITIALIZATION,
        kernel_regularizer=HYPER.REGULARIZER,
    )(X_st)
    
    if HYPER.BATCH_NORMALIZATION:
        X_st = tf.keras.layers.BatchNormalization()(X_st)


    ### Create and join the encoders ###

    # create empty lists for joing layers and inputs of total prediction model
    input_list = []
    joining_list = []


    ### create X_t encoder ###
    X_t_encoder = tf.keras.Model(inputs=X_t_input, outputs=X_t)
    input_list.append(X_t_input)
    joining_list.append(X_t)


    ### create X_s1 encoder ###
    X_s1_encoder = tf.keras.Model(inputs=X_s1_input, outputs=X_s1)
    input_list.append(X_s1_input)
    joining_list.append(X_s1)


    ### create X_st encoder ###
    X_st_encoder = tf.keras.Model(inputs=X_st_input, outputs=X_st)
    input_list.append(X_st_input)
    joining_list.append(X_st)


    ### create joint encoder ###
    joining_layer = tf.keras.layers.concatenate(joining_list)
    
    for i in range(HYPER.ENCODER_LAYERS):
        joining_layer = tf.keras.layers.Dense(
            HYPER.NODES_PER_LAYER_DENSE,
            activation=HYPER.DENSE_ACTIVATION,
            kernel_initializer=HYPER.INITIALIZATION,
            kernel_regularizer=HYPER.REGULARIZER,
        )(joining_layer)
        
        if HYPER.BATCH_NORMALIZATION:
            joining_layer = tf.keras.layers.BatchNormalization()(joining_layer)

    joining_layer = tf.keras.layers.Dense(
        HYPER.ENCODING_NODES_X_joint,
        activation=HYPER.ENCODING_ACTIVATION,
        kernel_initializer=HYPER.INITIALIZATION,
        kernel_regularizer=HYPER.REGULARIZER,
    )(joining_layer)
    
    if HYPER.BATCH_NORMALIZATION:
        joining_layer = tf.keras.layers.BatchNormalization()(joining_layer)
        
    X_joint_encoder = tf.keras.Model(inputs=input_list, outputs=joining_layer)


    ### Create total prediction model ###

    for i in range(HYPER.NETWORK_LAYERS):
        joining_layer = tf.keras.layers.Dense(
            HYPER.NODES_PER_LAYER_DENSE,
            activation=HYPER.DENSE_ACTIVATION,
            kernel_initializer=HYPER.INITIALIZATION,
            kernel_regularizer=HYPER.REGULARIZER,
        )(joining_layer)
        
        if HYPER.BATCH_NORMALIZATION:
            joining_layer = tf.keras.layers.BatchNormalization()(joining_layer)

    if HYPER.PROBLEM_TYPE == 'regression':
        consumption_output = tf.keras.layers.Dense(
            len(Y_example),
            activation='softplus',
            kernel_initializer=HYPER.INITIALIZATION,
        )(joining_layer)
        
    elif HYPER.PROBLEM_TYPE == 'classification':
        consumption_output = tf.keras.layers.Dense(
            len(Y_example) * HYPER.REGRESSION_CLASSES,
            kernel_initializer=HYPER.INITIALIZATION,
        )(joining_layer)
        
        consumption_output = tf.keras.layers.Reshape(
            (len(Y_example), HYPER.REGRESSION_CLASSES)
        )(consumption_output)
        
        consumption_output = tf.keras.activations.softmax(
            consumption_output, 
            axis=2
        )

    # create the tf model and define its inputs and outputs
    prediction_model = tf.keras.Model(
        inputs=input_list, 
        outputs=consumption_output
    )

    # create class instance for encoding and prediction models
    models = EncodersAndPredictor(
        X_t_encoder, 
        X_s1_encoder, 
        X_st_encoder, 
        X_joint_encoder, 
        prediction_model
    )

    # give us the summary of the total prediction model that we define
    prediction_model.summary()

    # visualize the encoding model and prediction model graphs
    if plot:

        for model_name, tf_model in models.__dict__.items():
            print('Computation graph for ' + model_name + ':')
            model_name = 'images/' + model_name + '.png'
            display(
                tf.keras.utils.plot_model(
                    tf_model, 
                    model_name, 
                    show_shapes=True
                )
            )
            
            print('---' * 35)

    # return model class instance
    return models


def train_model(
    HYPER,
    model,
    train_data,
    val_data,
    raw_data,
    loss_object,
    optimizer,
    mean_loss,
    monitor='val_loss',
    silent=True,
    plot=False,
):

    """ Trains and validates the passed prediction model with the passed 
    train_data and val_data datasets. Returns the train loss and validation loss 
    histories as numpy arrays.
    """

    if silent:
        verbose = 0
    else:
        verbose = 1
        

    ###
    # Define training and testing steps for functional API ###
    ###

    if HYPER.PROBLEM_TYPE == 'regression':

        # keep training and validation labels. Create copies so as to not change 
	      # original classes.
        train_data.Y_copy = train_data.Y
        val_data.Y_copy = val_data.Y

        # define the training step to execute in each iteration with magic
        @tf.function
        def train_step(model_input_list, Y_data):
        
            with tf.GradientTape() as tape:
            
                predictions = model(model_input_list, training=True)
                loss = loss_object(predictions, Y_data)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            mean_loss(loss)

            return loss

        # define the test step to execute in each iteration with a magic handle
        @tf.function
        def test_step(model_input_list, Y_data):
        
            predictions = model(model_input_list, training=False)
            t_loss = loss_object(predictions, Y_data)
            mean_loss(t_loss)

    elif HYPER.PROBLEM_TYPE == 'classification':

        train_data.Y_copy = train_data.Y
        val_data.Y_copy = val_data.Y

        # convert training and validation labels into classes. Create copies so 
	      # as to not  change original classes.
        train_data.Y = np.around(
            (train_data.Y - raw_data.Y_min)
            / raw_data.Y_range
            * HYPER.REGRESSION_CLASSES
        )
        
        val_data.Y = np.around(
            (val_data.Y - raw_data.Y_min) / raw_data.Y_range * HYPER.REGRESSION_CLASSES
        )

        # define the training step to execute in each iteration with magic 
        @tf.function
        def train_step(model_input_list, Y_data):
        
            with tf.GradientTape() as tape:
            
                predictions = model(model_input_list, training=True)
                loss = 0
                
                for i in range(HYPER.PREDICTION_WINDOW):
                
                    prediction = predictions[:, i, :]
                    loss += loss_object(Y_data[:, i], prediction)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            mean_loss(loss)

            return loss

        # define the test step to execute in each iteration with a magic handle
        @tf.function
        def test_step(model_input_list, Y_data):
        
            predictions = model(model_input_list, training=False)
            t_loss = 0
            
            for i in range(HYPER.PREDICTION_WINDOW):
            
                prediction = predictions[:, i, :]
                t_loss += loss_object(Y_data[:, i], prediction)

            mean_loss(t_loss)

    ###
    # Define how to batch data in each training step ###
    ###

    if HYPER.SPATIAL_FEATURES == 'image':

        def create_batched_data(dataset, batching_steps):

            """ """

            # build batches of data
            for j in range(batching_steps):

                # Get training data of currently iterated batch
                x_t = dataset.X_t[i + j]
                x_st = dataset.X_st[i + j]
                y = dataset.Y[i + j]
                building_id = dataset.X_s[i + j][0]
                cluster_id = dataset.X_s[i + j][1]

                # Prepare imagery data
                x_s1 = raw_data.building_imagery_data_list[
                    raw_data.building_imagery_id_list.index(int(building_id))
                ]

                # Expand dimensions for batching
                x_t = np.expand_dims(x_t, axis=0)
                x_s1 = np.expand_dims(x_s1, axis=0)
                x_st = np.expand_dims(x_st, axis=0)
                y = np.expand_dims(y, axis=0)

                # Create batches
                if j == 0:

                    X_t_batched = x_t
                    X_s1_batched = x_s1
                    X_st_batched = x_st
                    Y_batched = y

                else:

                    X_t_batched = np.concatenate((X_t_batched, x_t), axis=0)
                    X_s1_batched = np.concatenate((X_s1_batched, x_s1), axis=0)
                    X_st_batched = np.concatenate((X_st_batched, x_st), axis=0)
                    Y_batched = np.concatenate((Y_batched, y), axis=0)

            # Create model input list
            model_input_list = [X_t_batched, X_s1_batched, X_st_batched]

            return model_input_list, Y_batched

    else:

        def create_batched_data(dataset, batching_steps):

            """ """

            # build batches of data
            for j in range(batching_steps):

                # Get training data of currently iterated batch
                x_t = dataset.X_t[i + j]
                x_st = dataset.X_st[i + j]
                x_s1 = dataset.X_s1[i + j]
                y = dataset.Y[i + j]

                # Expand dimensions for batching
                x_t = np.expand_dims(x_t, axis=0)
                x_s1 = np.expand_dims(x_s1, axis=0)
                x_st = np.expand_dims(x_st, axis=0)
                y = np.expand_dims(y, axis=0)

                # Create batches
                if j == 0:

                    X_t_batched = x_t
                    X_s1_batched = x_s1
                    X_st_batched = x_st
                    Y_batched = y

                else:

                    X_t_batched = np.concatenate((X_t_batched, x_t), axis=0)
                    X_s1_batched = np.concatenate((X_s1_batched, x_s1), axis=0)
                    X_st_batched = np.concatenate((X_st_batched, x_st), axis=0)
                    Y_batched = np.concatenate((Y_batched, y), axis=0)

            # Create model input list
            model_input_list = [X_t_batched, X_s1_batched, X_st_batched]

            return model_input_list, Y_batched

    ###
    # Perform epochs of training and validation ###
    ###

    # create an empty lists to save training and validation loss results for each epoch
    val_loss_history = []
    train_loss_history = []

    for epoch in range(HYPER.EPOCHS):

        if not silent:
        
            # tell which epoch we are at
            print('Epoch {}/{}'.format(epoch + 1, HYPER.EPOCHS))


        ###
        # Training ###
        ###

        # Reset the metrics at the start of the next epoch
        mean_loss.reset_states()
        
        # Shuffle training data
        train_data.randomize()

        if not silent:
        
            # tell us that we start training now
            print('Training:')

            # create a progress bar for training
            progbar = tf.keras.utils.Progbar(
                math.floor(train_data.n_datapoints - HYPER.BATCH_SIZE)
                / HYPER.BATCH_SIZE,
                stateful_metrics=['loss'],
            )

        # iterate over training data in BATCH_SIZE steps
        for i in range(
            0, 
            train_data.n_datapoints - HYPER.BATCH_SIZE, 
            HYPER.BATCH_SIZE
        ):

            # call function to create batched model inputs and labels
            model_input_list, Y_batched = create_batched_data(
                train_data, HYPER.BATCH_SIZE
            )

            # Execute the training step for this batch
            train_step(model_input_list, Y_batched)

            # update the progress bar
            if not silent:
            
                values = [('loss', mean_loss.result().numpy())]
                progbar.add(1, values=values)

        # add training loss to history
        train_loss_history.append(mean_loss.result().numpy())


        ###
        # Validation ###
        ###

        # Reset the metrics at the start of the next epoch
        mean_loss.reset_states()

        # Shuffle validation data
        val_data.randomize()

        if not silent:
        
            # Tell us that we start validating now
            print('Validation:')

            # create a progress bar for validation
            progbar = tf.keras.utils.Progbar(
                math.floor(val_data.n_datapoints - HYPER.BATCH_SIZE) / HYPER.BATCH_SIZE,
                stateful_metrics=['loss'],
            )

        # iterate over validation data in BATCH_SIZE steps
        for i in range(0, val_data.n_datapoints - HYPER.BATCH_SIZE, HYPER.BATCH_SIZE):

            # call function to create batched model inputs and labels
            model_input_list, Y_batched = create_batched_data(
                val_data, HYPER.BATCH_SIZE
            )

            # Execute the testing step for this batch
            test_step(model_input_list, Y_batched)

            # update the progress bar
            if not silent:
            
                values = [('loss', mean_loss.result().numpy())]
                progbar.add(1, values=values)

        # add validation loss to history
        val_loss_history.append(mean_loss.result().numpy())


        ###
        # implement early break here ###
        ###

        if monitor == 'val_loss':
            current_minimum = min(val_loss_history[-HYPER.PATIENCE :])
            
        elif monitor == 'loss':
            current_minimum = min(train_loss_history[-HYPER.PATIENCE :])
            
        elif monitor is None:
            continue

        if epoch == 0:
            total_minimum = current_minimum
            
        if current_minimum > total_minimum:
            break
            
        else:
            total_minimum = min(total_minimum, current_minimum)

    if HYPER.PROBLEM_TYPE == 'classification':
    
        train_data.Y = train_data.Y_copy
        val_data.Y = val_data.Y_copy

    # Plot training and validation history
    if plot:

        plt.figure(figsize=(16, 8))
        plt.plot(train_loss_history)
        plt.plot(val_loss_history)
        plt.title('Training and validation loss history of neural network')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()
        
        
    return train_loss_history, val_loss_history


def test_model(
    HYPER,
    figtitle,
    model,
    test_data,
    raw_data,
    mean_loss,
    loss_function,
    silent=True,
    plot=False,
):

    """ Makes predictions on passed test_data using the past model, and returns 
    the calculated testing loss. Predictions are saved under the passed Dataset 
    object's attribute called predictions.
    """

    # Reset the state of the test loss metric
    mean_loss.reset_states()

    if HYPER.SPATIAL_FEATURES == 'image':

        # create a zero matrix for saving prediction results.
        if HYPER.PROBLEM_TYPE == 'regression':
        
            predictions = np.zeros(
                (
                    test_data.n_datapoints, 
                    HYPER.PREDICTION_WINDOW
                )
            )
            
        elif HYPER.PROBLEM_TYPE == 'classification':
        
            predictions = np.zeros(
                (
                    test_data.n_datapoints,
                    HYPER.PREDICTION_WINDOW,
                    HYPER.REGRESSION_CLASSES,
                )
            )

        # iterate over all testing data
        for i in range(test_data.n_datapoints):

            # Get training data of currently iterated batch #
            x_t = test_data.X_t[i]
            x_st = test_data.X_st[i]
            y = test_data.Y[i]
            building_id = test_data.X_s[i][0]
            cluster_id = test_data.X_s[i][1]

            ## Prepare imagery data
            x_s1 = raw_data.building_imagery_data_list[
                raw_data.building_imagery_id_list.index(int(building_id))
            ]

            # Expand dimensions for batching
            x_t = np.expand_dims(x_t, axis=0)
            x_s1 = np.expand_dims(x_s1, axis=0)
            x_st = np.expand_dims(x_st, axis=0)
            y = np.expand_dims(y, axis=0)

            # Create model input list
            model_input_list = [x_t, x_s1, x_st]

            # make predictions and save results in respective matrix
            predictions[i] = model.predict(model_input_list)

    else:

        # make predictions
        predictions = model.predict(
            [
                test_data.X_t, 
                test_data.X_s1, 
                test_data.X_st
            ]
        )

    ###
    # Calculate the testing loss ###
    ###

    # convert predictions back from classes to floating point values
    if HYPER.PROBLEM_TYPE == 'classification':
    
        predictions = np.argmax(predictions, axis=2)
        predictions = (
            predictions * raw_data.Y_range / (
                HYPER.REGRESSION_CLASSES 
                + raw_data.Y_min
            )
        )

    # calculate the testing losses
    t_loss = loss_function(test_data.Y, predictions)

    # take the mean of single losses
    testing_loss = mean_loss(t_loss).numpy()

    # tell us how much testing loss we have
    if not silent:
    
        print(figtitle + ' loss:', testing_loss)

    test_data.predictions = predictions
    test_data.testing_loss = testing_loss

    # Plot exemplar predictions
    if plot:
    
        plot_true_vs_prediction(figtitle, test_data.Y, predictions)

    return testing_loss
