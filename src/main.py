
### Package imports and computation environment setup ###

# Import existing packages
import gc
import os
import random
import math
import numpy as np
import tensorflow as tf

# Import own application source code
import hyperparameters
import data
import prediction
import activelearning
import addexperiments
import saveresults

# Set a randomization seed for better reproducability of results
seed = 3
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# class instance that contains all our hyperparameters
HYPER = hyperparameters.HyperParameter(random_seed=seed)


### 1. Data preprocessing ###

# initialize raw_data
raw_data = data.RawData(HYPER)

# 1.1 Building-scale electric consumption profiles
raw_data = data.import_consumption_profiles(HYPER, raw_data, plot=False)

# 1.2 Building-scale aerial imagery
raw_data = data.import_building_images(HYPER, raw_data, plot=False)

# 1.3 Cluster-scale meteorological data
raw_data = data.import_meteo_data(HYPER, raw_data, plot=False)

# 1.4 Feature-label pairing
dataset, raw_data = data.create_feature_label_pairs(HYPER, raw_data)

# 1.5 Encode temporal features
dataset = data.encode_time_features(HYPER, dataset, silent=True)

# 1.6 Normalize all features
dataset = data.normalize_features(HYPER, raw_data, dataset, silent=True)

# 1.7 Split into training, validation and testing data
(
    training_data, 
    validation_data, 
    testing_data
) = data.split_train_val_test(HYPER, raw_data, dataset)

# 1.8 Standardize features
testing_data = data.standardize_features(
    HYPER, 
    raw_data, 
    testing_data, 
    training_data, 
    silent=True
)
validation_data = data.standardize_features(
    HYPER, 
    raw_data, 
    validation_data, 
    training_data, 
    silent=True
)
training_data = data.standardize_features(
    HYPER, 
    raw_data, 
    training_data, 
    training_data, 
    silent=True
)


### 2. Prediction model ###

# initialize optimization parameters
(
    loss_object, 
    optimizer, 
    loss_function, 
    mean_loss
) = prediction.initialize_optimizer(HYPER)


### 2.1 Baseline ###

# train a random forest model
RF_regr = prediction.create_and_train_RF(HYPER, training_data)

# make predictions
train_pred = prediction.predict_with_RF(
    HYPER, 
    RF_regr, 
    training_data
)
val_pred = prediction.predict_with_RF(
    HYPER, 
    RF_regr, 
    validation_data
)
test_pred = prediction.predict_with_RF(
    HYPER, 
    RF_regr, 
    testing_data
)
    
# Calculate the loss on each prediction
mean_loss.reset_states()
train_l = mean_loss(
    loss_function(
        training_data.Y, 
        train_pred
    )
).numpy()

mean_loss.reset_states()
val_l = mean_loss(
    loss_function(
        validation_data.Y, 
        val_pred
    )
).numpy()

mean_loss.reset_states()
test_l = mean_loss(
    loss_function(
        testing_data.Y, 
        test_pred
    )
).numpy()

RF_result = test_l
    
# Tell us the out of bag validation score and prediction losses
print(
    'The out-of-bag validation score for random forest is:', 
    RF_regr.oob_score_
)
print(
    'Loss on training data:             {}'.format(
        train_l
    )
)
print(
    'Loss on validation data:           {}'.format(
        val_l
    )
)
print(
    'Loss on test data:         {}'.format(
        test_l
    )
)

# delete the RF model as it occupies memory
del RF_regr
_ = gc.collect()


### 2.2 Definition ###

models = prediction.build_prediction_model(
    HYPER, 
    raw_data, 
    training_data, 
    plot=False
)


### 2.3 Training ###

train_hist, val_hist = prediction.train_model(
    HYPER, 
    models.prediction_model, 
    training_data, 
    validation_data, 
    raw_data,
    loss_object, 
    optimizer, 
    mean_loss,
    silent=False
)

prediction.save_prediction_model(
    HYPER, 
    raw_data, 
    models.prediction_model, 
    'initial'
)
prediction.save_encoder_and_predictor_weights(
    HYPER, 
    raw_data, 
    models
)


### 3. Active learning ###

# initialize hyper parameters for AL
HYPER.set_act_lrn_params()


### 3.4 Batching algorithm ###
    
# create random result for benchmark once only for this pred_type
PL_result =  activelearning.feature_embedding_AL(
    HYPER, 
    models, 
    raw_data, 
    training_data, 
    testing_data, 
    loss_object, 
    optimizer, 
    mean_loss,
    loss_function,
    'PL', 
    silent=False
)

# create empty list for saving results of corresponding AL variable
var_result_dict = {}

# iterate over all sort variables that are chosen to be considered
for query_variable in HYPER.QUERY_VARIABLES_ACT_LRN:

    # empty list for savings results of correspnding AL variant
    method_result_dict = {}

    # iterate over all methods that are chosen to be considered
    for method in HYPER.QUERY_VARIANTS_ACT_LRN:

        if HYPER.TEST_EXPERIMENT_CHOICE == 'main_experiments':
            AL_result = activelearning.feature_embedding_AL(
                HYPER, 
                models, 
                raw_data, 
                training_data, 
                testing_data,
                loss_object, 
                optimizer, 
                mean_loss,
                loss_function,
                method, 
                AL_variable=query_variable, 
                silent=False
            )
            
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'sequence_importance':
            AL_result = addexperiments.test_sequence_importance_AL(
                HYPER, 
                models, 
                raw_data, 
                training_data, 
                testing_data, 
                loss_object, 
                optimizer, 
                mean_loss,
                loss_function, 
                method, 
                AL_variable=query_variable,
                silent=False
            )
            
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'subsample_importance':
            AL_result = addexperiments.test_subsample_importance_AL(
                HYPER, 
                models, 
                raw_data, 
                training_data, 
                testing_data, 
                loss_object, 
                optimizer, 
                mean_loss,
                loss_function, 
                method, 
                AL_variable=query_variable,
                silent=False
            )
            
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'pointspercluster_importance':
            AL_result = addexperiments.test_pointspercluster_importance_AL(
                HYPER, 
                models, 
                raw_data, 
                training_data, 
                testing_data, 
                loss_object, 
                optimizer, 
                mean_loss,
                loss_function, 
                method, 
                AL_variable=query_variable,
                silent=False
            )
            
        elif HYPER.TEST_EXPERIMENT_CHOICE == 'querybycoordinate_importance':
            AL_result = addexperiments.test_querybycoordinate_importance_AL(
                HYPER, 
                models, 
                raw_data, 
                training_data, 
                testing_data, 
                loss_object, 
                optimizer, 
                mean_loss,
                loss_function, 
                method, 
                AL_variable=query_variable,
                silent=False
            )
            
        # add results to method_result_list
        method_result_dict[method] = AL_result
     
    # add results to var_result_list
    var_result_dict[query_variable] = method_result_dict


results_dict = {
    'RF_result' : RF_result,
    'PL_result' : PL_result,
    'AL_result' : var_result_dict
}

# save results
saveresults.saveallresults(
    HYPER,
    raw_data,
    results_dict
)

