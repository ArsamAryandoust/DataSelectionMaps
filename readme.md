# Enhanced spatio-temporal electric load forecasts with less data using active deep learning
---


## Content overview 

1. Data processing
2. Prediction model
3. Active learning

An effective way to mitigate climate change is to electrify our energy sectors, and supply their electricity from renewable wind and solar, which are highly fluctuating and uncertain sources of energy. Planning and operating electricity grids under these uncertainties increasingly requires fine-grained and accurate predictions of electric load across very short to long time windows. Among the different types of electric load forecasts that are performed, spatio-temporal predictions have gained increasing importance; they predict load for times and places, in which we do not have detailed information about electric load profiles in our grids, and operate these as black boxes. 

While remotely sensed data from e.g. satellite imagery is increasingly easy to access for making spatio-temporal predictions, ground truth electric load data remains difficult and expensive to collect. One reason for this is that electric utilities are limited in the number of smart meters they can place to collect consumption data by social, financial and technical barriers. Another reason is that utilities are further limited in the amount of data they can query from each smart meter in real-time, something known as the velocity constraint of data, by data communication bandwidths and privacy concerns of consumers. The figure below shows the current state of smart meter adoption around the globe. For regions with an already high adoption of meters like in Scandinavian countries or Spain, we must know when to query data from which smart meter so as to make the best possible predictions of electric load, without exceeding our data velocity constraints. For regions with a yet low adoption of smart meters like in most of Africa, South and Latin America or Asia, we additionally want to know where to place new smart meters first so as to make the best possible predictions of electric load for yet unsensed parts of our grids, while effectively using our budget for installing new smart meters. 

<img src="images/markdown/figure1.png">


Given the aerial image of a building, the meteorological conditions in the region of that building and a time stamp, we want to predict the electric load profile of the building for the next 24 hours in 15 min steps. We start with a prediction model that has learnt this relationship for a few buildings and times. Our features are all remotely sensed and assumed to be available for every building and point in time at no cost. For every new load profile that we collect, we assume to experience some cost and are constrained in the total number of profiles that we can collect by some budget $n_{budget}$. Our goal is to collect further ground truth data, i.e. the electric load profiles at different times and buildings, so as to make the best possible predictions for buildings and times, for which we do not have load profiles available, without exceeding $n_{budget}$. We use load profiles from two different datasets: one containing 100 buildings, and one cointaining 400 buildings in Switzerland. In each experiment, we randomly select load profiles from 800 time stamps in 2014 to create the candidate data pool for our experiments. The figure below visualizes the modular DL prediction model architecture that we propose for solving this task; we call it embedding network. 

<img src="images/markdown/figure2.png">

In each iteration of the active deep learning (ADL) algorithm that we apply, we query a batch of candidate data points. We remove queried data from the candidate data pool at a rate $\delta$. Keeping queried points, i.e. allowing the algorithm to re-use them, lets us further reduce data usage. First, we encode the features of candidate data points into an embedded vector space using a modular neural network prediction model that is trained on initially available data points. We then cluster candidate data points based on their vector distances to each other in this encoded space, with the number of clusters being equal to our query batch size. Next, we calculate the distance of the vector of each encoded data point to its cluster center, and query one data point per cluster based on these distances. We test our ADL method for randomized, maximized, minimized and averaged distances of embedded data points to their cluster centers in every queried data batch. We refer to these as our ADL variants. 

The figure bellow visualizes the differences between data queries with each of our ADL variants. In a first variant of our ADL method, we randomly select data points from each embedded cluster of candidates (a.). In a second variant, we query candidate data points whose embedded feature vectors are further away from their cluster centers (b.). We expect to be more uncertain about these points, as they are more likely to be true members of another cluster: we likely explore the data that is close to our decision boundaries, if not outliers, and expect a larger surprise/learning experience from querying labels for these data points. In a third variant, we query labels of data points that are close to their cluster centers, which we expect to be more representative of their clusters and respectively our entire data population (c.). In a fourth variant, we query data points that have the largest distance to the average of distances to cluster centers among all points of the same cluster, which results in a combination of queries alternating between uncertain and representative data points (d.). Each of these ADL variants tries to select a subset of data points from the candidate data pool in a different way that is more informative compared to when selecting these uniformly at random using passive deep learning (PDL). The distance of candidate data points to their cluster centers in an embedded vector space is a new metric of information that we propose; we call it embedding uncertainty. 

<img src="images/markdown/figure3.png">

We evaluate the performance of our ADL and PDL algorithms for spatial, temporal and spatio-temporal predictions compared to a random forest (RF) benchmark. In this context, temporal predictions mean that we predict the load profile for buildings in which a sensor is placed, but for a time period into the past or future, for which we do not have measured data available; this allows us to test performance against a distribution shift of our data in time. Spatial predictions mean that we predict a load profile for buildings in which a sensor is not placed, but for a time period in which we do have load profiles available for other buildings; this allows us to test performance against a distribution shift of our data in space. Spatio-temporal predictions respectively refer to the most difficult problem of predicting load profiles for times and buildings, for which we do not have any load profiles available at all; this allows us to test performance against a distribution shift of our data in both time and space. We refer to these as the different prediction types that we evaluate. 

For each prediction type that we evaluate, we further distinguish between the type of features that we can encode for querying candidate data points. We distinguish between features that are variant in time $\mathbf{x}_t$ (time stamp), space $\mathbf{x}_s$ (building image), and both time and space $\mathbf{x}_{st}$ (meteorological data), as well as the entire feature vector $\mathbf{x}_{t,s}$ which is concatenated from these three vectors.  As the predicted output of our network $\mathbf{\hat{y}}_{t,s}$ represents a vector, i.e. the electric consumption of a building for the next 24 hours in 15-min steps (96 values), we can use this vector as an embedding of our entire feature vector $\mathbf{x}_{t,s}$. In a further test, we hypothetically use our true labels $\mathbf{y}_{t,s}$ for querying candidate data points so as to see how our proposed metric and ADL variants perform with optimal priors on the distances of labels that we try to infer from our embedded feature spaces. We refer to these as our ADL variables. 

Our findings can have important implications for the energy transition to a carbon free power systems and therefore for mitigating climate change. The figure below shows how the ADL method we propose can be used by distribution and transmission system operators for electricity to make overall more accurate predictions of electric load using budgets for installing smart meters and streaming their data more effectively. We start with choosing a set of locations to place sensors uniformly at random (a.). Then, we collect training data for making spatio-temporal predictions and learn where to best place a relatively large set of new sensors next (b.). Next, we iteratively place fewer sensors to make better predictions in the most informative sequence (c. - f.).

<img src="images/markdown/figure5.png">

---

## File overview

* src: folder containing the source code for jupyter notebooks and python scripts
* tests: folder containing unit tests for source code
* data: folder containing required data for running experiments
* results: folder containing results after running experiments
* images: folder containing images for this readme and the manuscript
* Docker: repository containing a Dockerfile for running hypothesis test inside a docker container
* DockerNotebook: repository containing a Dockerfile for running notebooks inside a docker container
* DockerUnittests: repository containing a Dockerfile for running unit tests inside a docker container
* src/main.py: main file for reproducing experimental results
* src/hyperparameters.py: module for choosing all hyper parameters of experiments
* src/data.py: module for processing data 
* src/prediction.py: module for building, training and testing prediction models
* src/activelearning.py: active learning methods
* src/main_notebook.ipynb: Jupyter notebook session for reproducing method and experiments
* src/results_notebook.ipynb: Jupyer notebook sessions for plotting and analysing experimental results
* tests/test_prediction.py: unit tests for prediction.py
* tests/test_activelearning.py: unit tests for activelearning.py
---

## Docker 

Download this repository to your home directory, change your current working directory to this repository, and build main docker container by running:

```
cd
git clone https://github.com/ArsamAryandoust/DataSelectionMaps
cd DataSelectionMaps
docker build -t main Docker
```
For CPU computation, run:

`docker run -v ~/DataSelectionMaps:/DataSelectionMaps main`

For GPU support, make sure you have the nvidia-container-toolkit installed, then run:

`docker run --gpus all -v ~/DataSelectionMaps:/DataSelectionMaps main`

**Note:** If you download this repository to a location other than your home directory, make sure it is at a sub-directory of your home directory and replace all ~/ActiveLearning with ~/path/to/subdirectory/ActiveLearning in the commands, i.e.

`docker run -v ~/path/to/subdirectory/DataSelectionMaps:/DataSelectionMaps main`


`docker run --gpus all -v ~/path/to/subdirectory/DataSelectionMaps:/DataSelectionMaps main`

---

## Jupyter notebooks with docker

For running jupyter notebook sessions inside a docker container, run:

`docker build -t main_notebook DockerNotebook`

For CPU computation, run:

`docker run -it -v ~/DataSelectionMaps:/DataSelectionMaps -p 3333:1111 main_notebook`

For GPU support, make sure you have the nvidia-container-toolkit installed, and run:

`docker run -it --gpus all -v ~/DataSelectionMaps:/DataSelectionMaps -p 3333:1111 main_notebook`

Next, open localhost:3333 in your browser and type in the token from your terminal as password to run Jupyter notebook sessions inside the docker container from your browser.

**Note:** If you download this repository to a location other than your home directory, make sure it is at a sub-directory of your home directory and replace all ~/ActiveLearning with ~/path/to/subdirectory/ActiveLearning in the commands, i.e.

`docker run -it -v ~/path/to/subdirectory/DataSelectionMaps:/DataSelectionMaps -p 3333:1111 main_notebook`


`docker run -it --gpus all -v ~/path/to/subdirectory/DataSelectionMaps:/DataSelectionMaps -p 3333:1111 main_notebook`

---

## Unit tests with docker

For running unittests for the implemented active learning algorithm inside a docker container, run:

`docker build -t tests DockerUnittests`

For CPU computation, run:

`docker run -v ~/DataSelectionMaps:/DataSelectionMaps tests`

For GPU support, make sure you have the nvidia-container-toolkit installed, then run:

`docker run --gpus all -v ~/DataSelectionMaps:/DataSelectionMaps tests`

**Note:** If you download this repository to a location other than your home directory, make sure it is at a sub-directory of your home directory and replace all ~/ActiveLearning with ~/path/to/subdirectory/ActiveLearning in the commands, i.e.

`docker run -v ~/path/to/subdirectory/DataSelectionMaps:/DataSelectionMaps tests`


`docker run --gpus all -v ~/path/to/subdirectory/DataSelectionMaps:/DataSelectionMaps tests`

---

We start with importing a number of packages that we need for implementing this algorithm, and define a number of hyper parameters that are bundled through the use of attributes of a class object *HyperParameter* defined in **hyperparameters.py**.


```python
### Package imports and computation environment setup ###

import gc
import os
import random
import math
import numpy as np

# decide which GPU(s) should be seen by CUDA before importing tf.
# Choose '-1' for calculation on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

# tell us how many GPUS are visible to CUDA and tensorflow
print(
    'Num GPUs Available: ', 
    len(tf.config.experimental.list_physical_devices('GPU'))
)

# Import application source code
import hyperparameters
import data
import prediction
import activelearning


# Set a randomization seed for better reproducability of results,
# or None for avoiding any possible effect on experiments
seed = 3

if seed is not None:

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Set seed for tf weight initialzers
    random_normal = tf.keras.initializers.RandomNormal(
        mean=0.0, 
        stddev=0.05, 
        seed=seed
    )
    random_uniform = tf.keras.initializers.RandomUniform(
        minval=-0.05, 
        maxval=0.05, 
        seed=seed
    )
    truncated_normal = tf.keras.initializers.TruncatedNormal(
        mean=0.0, 
        stddev=0.05, 
        seed=seed
    )
    glorot_normal = tf.keras.initializers.GlorotNormal(
        seed=seed
    )
    glorot_uniform = tf.keras.initializers.GlorotUniform(
        seed=seed
    )
    orthogonal = tf.keras.initializers.Orthogonal(
        gain=1.0, 
        seed=seed
    )
    variance_scaling = tf.keras.initializers.VarianceScaling(
                       scale=1.0, 
                       mode='fan_in', 
                       distribution='truncated_normal', 
                       seed=seed
    )

# class instance that contains all our hyperparameters
HYPER = hyperparameters.HyperParameter(random_seed=seed)
```

    Num GPUs Available:  1
    
     You run in public access mode, and only have access to public data. HYPER.LABELS is set to "feature_scaled".


## 1 Data processing

In this section, we create training, validation and testing datasets which are ready to be processed by the deep learning models that we will create in the next section. First, we import our underlying label and feature data. Our labels are building-scale electric consumption profiles. Our features are high resolution aerial images of these buildings and the meteorlogical conditions at these buildings. Next, we build feature label pairs from these raw data and further add the timestamps at which electric consumption values are measured as another feature to each of our data points. We then proceed with normalizing our data, splitting it for training, validation and testing, and eventually standardizing all datasets with the statistics (mean and standard deviation of features) of the training data. 

We start with creating two new class objects: first, *RawData*, which we use to bundle paths to where our data is stored and miscellaneous other useful values; second, *Dataset*, which we use to bundle vectors of time variant (X_t), space variant (X_s, X_s1) and space-time variant (X_st) features, as well as their corresponding labels (Y). Classes and functions that we introduce in this section are defined in **data.py**.


```python
raw_data = data.RawData(HYPER)
```

### 1.1 Building-scale electric consumption profiles

The electric consumption profiles that we use are measured in 15-min time steps. Each day hence has 96 values. Each year respectively has 35,040 values. The profiles are mostly added up from single sub profiles, so as to eventually represent the total consumption of the corresponding building, and not only of distinct entities inside each building, such as several apartments inside the same building. Each building and its corresponding load profile is identified through a building ID, a cluster ID and a year ID which we import and save separately for all of these profiles. The plots below show the profiles of exmeplar buildings. The x-axes represent the timestamp in which the electric consumption is measured, and the y-axes the electric power consumption in kilo Watt (hours) [kW/kWh]. 


```python
raw_data = data.import_consumption_profiles(HYPER, raw_data)
```

    Importing consumption profiles
    1/1 [==============================] - 1s 561ms/step
    The 92 imported electric load profiles demand a total amount of 0.02606976 GB of RAM



    
![png](images/markdown/output_5_1.png)
    


### 1.2 Building-scale aerial imagery

For each building whose electric consumption profile we loaded, we further import an aerial image that is cropped to an excerpt from that building only. Images are padded with zero values (black) on their empty borders so as to all have the same size. Here, we also perform the optional down sampling and grey scaling of images. For each imported building image, we further keep track of their corresponding building IDs so as to be able to assign these to each other during further data processing or during training, validation and testing of our prediction model.


```python
raw_data = data.import_building_images(HYPER, raw_data)
```

    Importing building-scale aerial imagery:
    92/92 [==============================] - 0s 56us/step
    The 92 aerial images demand 0.00022080000000000024 GB RAM with float32 entries


### 1.3 Cluster-scale meteorological data

Buildings that are in a maximum distance of one kilo meter (km) to each other belong to the same cluster. Instead of using meteorological data for each individual building, we use meteorological data for each cluster. This significantly reduces our data demand for meteorological data without any relevant loss of accuracy. Here, we import the meteorological data for each cluster and keep track of the cluster IDs for each imported dataset. The resolution of these measurements is hourly. The number of past data points that we use for each data point is determined through hyper parameters. The plot below shows all meteorological conditions that we import and consider as features for an exemplar cluster ID. These are air density, cloud cover, precipitation, solar irradiance on the surface, solar irradiance on the atmosphere, snow mass, snowfall, temperature and wind speed. The x-axes represents the timestamp in which the meteorological conditions are measured, and the y-axes the respective values in their various respective units. These values consist of air density in kg/m³, cloud cover, precipitation in mm/hour, ground level solar irradiance in W/m², top of atmosphere solar irradiance W/m², air temperature in °C, snowfall in mm/hour, snow mass in kg/m² and wind speed.


```python
raw_data = data.import_meteo_data(HYPER, raw_data)
```

    Importing meteorological data
    11/11 [==============================] - 1s 44ms/step
    The 11 meteo data files demand 0.030835199999999993 GB RAM



    
![png](images/markdown/output_9_1.png)
    


### 1.4 Feature-label pairing

For each timestamp in which electric consumption and meteorological measurements match, and for each building, we create a unique data point consisting of features and labels. Here, we further add the timestamp distinguished by the year, month, day, hour and quarter hour as a feature to each data point. Therefore, each data point consists of a time variant feature vector X_t (timestamp), a space variant feature vector X_s (building ID), a space-time variant feature vector X_st (meteorological data) and a label vector Y (electric consumption). We bundle these as attributes of an object instance of the previously created *Dataset* class.

Note: At this stage we do not replicate images for each data point as this would unnecessarily increase our memory demand. Instead, we only save the building ID as spatial feature of each data point so as to be able to load images during training, without the need to replicate images multiple times for each data point. 


```python
dataset, raw_data = data.create_feature_label_pairs(HYPER, raw_data)
```

    Creating feature label data pairs:
    3312/3312 [==============================] - 2s 688us/step


### 1.5 Encode temporal features

Our time variant features consist of five dimensions: year, month, day, hour and quarter of hour. We need to express these features in more useful numbers. Here, we can do this through ordinal encoding for each dimension separately, or an ordinal encoding merged into a single dimension, or through one-hot-encoding. Depending on many other hyper parameters that we set, any of these can bring the best performance for predictions and the learning process of our neural networks. However, one-hot-encoded features have the largest and hence ideally avoided increase in memory demand. For our application, distinguished ordinal encoding of each dimension is a good choice.


```python
dataset = data.encode_time_features(HYPER, dataset)
```

    Encoding temporal features
    X_t before: [0. 1. 2. 1.]
    X_t after: [0 1 1 0] (ORD)


### 1.6 Normalize all features

It is often useful to normalize each feature to values between 0 and 1. This helps to prevent vanishing and exploding gradients when updating the weights of our neural network prediction model through backpropagation and gradient descent. The importance of this becomes particularly clear if we look at the different scales at which meteorological data is provided. Each meteorological condition has a distinct unit and hence a largely different scale than all other values, although being part of the same feature vector. Here, we normalize features by each of their feature dimensions, as if these were flattened feature vectors. 


```python
dataset = data.normalize_features(HYPER, raw_data, dataset)
```

    Normalizing features


### 1.7 Split into training, validation and testing data

Given our entire data, we want to split it into subsets for training, validation and testing of our prediction models. While models like random forest do not need a separate validation dataset, and can use the concept of out-of-bag error instead, the validation dataset is important when training neural network prediction models. At the same time, we cannot use the same validation dataset, which is used during training of our neural network to see whether generalization improves or not, as a reliable testing dataset, as we fit our neural network models such that they perform particularly well on the validation data through concepts like early stopping. Hence we somehow also fit our model to performing well on this validation data. Here we split our data according to the shares that are defined through our hyper parameters. The output below shows the shares into which we split our entire data. 

Note: We distinguish between temporal, spatial and spatio-temporal predictions. Temporal predictions mean that we want to predict the electric consumption for the same building for which a sensor is placed, but for a time period into the past or future, for which we do not have measured data available. Spatial predictions mean that we want to predict electric consumption for buildings in which a sensor is not placed, but for a time period in which we do have measurements available from other buildings. Spatio-temporal predictions respectively refer to the most difficult problem of predicting the electric consumption for times and buildings, for which we do not have any measurements available at all.


```python
(
    training_data, 
    validation_data, 
    spatial_test_data, 
    temporal_test_data, 
    spatemp_test_data
) = data.split_train_val_test(HYPER, raw_data, dataset)
```

    Splitting data into training, validation and testing sets.
    With TRAIN_SPLIT = 0.3  and TEST_SPLIT = 0.7 the data is split in the following ration:
    ------------------------------------------------------------------------------------------------------------------
    Training data:   88 (3%)
    Validation data: 203 (6%)
    Testing data:    3021 (91%)
    ------------------------------------------------------------------------------------------------------------------
    Spatial testing data:         701 (23%)
    Temporal testing data:        681 (23%)
    Spatio-temporal testing data: 1639 (54%)


### 1.8 Standardize features

We can only guarantee the convergence of our neural network model and speed up its training if we standardize our training data, meaning that we create a zero mean and unit variance for each of its distinct feature dimensions. We do this by first flattening all our input dimensions and then calculating the statistics that are required to perform this transformation. Next we perform the same transformation on all other validation and testing datasets with the statistics of our training data.

Note: All testing and validation datasets are transformed according to the statistics of the training data, not with respect to their own statistics as this would lead to differently transformed feature values and corrupt inference and validation. Furthermore, it is important that the statistics for performing standardization are calculated not on the entire data, but only based on the training data as only this is seen by the neural network during training and needs to be guaranteed to converge. 


```python
# Note: it is important that training data is
# converted last as it is the reference for the rest
spatemp_test_data = data.standardize_features(
    HYPER, 
    raw_data, 
    spatemp_test_data, 
    training_data, 
    silent=True
)
temporal_test_data = data.standardize_features(
    HYPER, 
    raw_data, 
    temporal_test_data, 
    training_data, 
    silent=True
)
spatial_test_data = data.standardize_features(
    HYPER, 
    raw_data, 
    spatial_test_data, 
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
```

## 2 Prediction model

In this section, we create baseline predictions using a random forest prediction model, and then create, train and test our embedding network prediction model. The embedding network model that we define is a modular neural network and each module is used as an encoder for embedding features for the subsequent active learning task. We start with defining a new class object *EncodersAndPredictors* for bundling our main prediction model and its encoder sub-networks. We further set the loss metric according to whether we solve our prediction task as a regression or a classification model. In the latter case we use a sparse categorical cross entropy loss instead of a mean square error. In both cases we choose to take the mean of each prediction step error as our total error, and a free but unpublished optimizer called *RMSprop* for training our weights. Classes and functions that we introduce in this section are defined in **prediction.py**.


```python
# initialize optimization parameters
(
    loss_object, 
    optimizer, 
    loss_function, 
    mean_loss
) = prediction.initialize_optimizer(HYPER)
```

### 2.1 Baseline

In order to see how well we predict with any chosen constellation of problem setup and neural network design, we create a random forest regressor as our baseline predictor. Random forest regressors are fast to train and powerful predictors at the same time. Another major advantage of this decision-tree based prediction model is that we do not have any hyper parameters to tune: the number and depth of trees can be as large as we can computationally afford, and the optimal number of randomly chosen features for creating each tree is known to be $\sqrt{D}$ (with theoretical guarantees), where $D$ is the dimension of our entire feature vector. As a rule of thumb, we can say that once we beat this predictor, we are on a good way to work ourselves forward to a useful deep learning prediction model. In addition to this, we can use the number of node activations of the random forest to make statements about the importance of features. The plots below illustrate the feature importance that are calculated in this way, and exemplar predictions for each available dataset. The x-axis of each plot describes the 15-min time steps into the future from a given timestamp, and the y-axis the corresponding electric consumption of a given building at these times in kW/kWh.

Note: The evaluation of predictions on our training dataset is not used for any validation, but to see whether our model learns useful patterns, or whether it under- or overfits to the training data. 


```python
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
spatial_test_pred = prediction.predict_with_RF(
    HYPER, 
    RF_regr, 
    spatial_test_data
)
temporal_test_pred = prediction.predict_with_RF(
    HYPER, 
    RF_regr, 
    temporal_test_data
)
spatemp_test_pred = prediction.predict_with_RF(
    HYPER, 
    RF_regr, 
    spatemp_test_data
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
test_l_spatial = mean_loss(
    loss_function(
        spatial_test_data.Y, 
        spatial_test_pred
    )
).numpy()

mean_loss.reset_states()
test_l_temporal = mean_loss(
    loss_function(
        temporal_test_data.Y, 
        temporal_test_pred
    )
).numpy()

mean_loss.reset_states()
test_l_spatemp = mean_loss(
    loss_function(
        spatemp_test_data.Y, 
        spatemp_test_pred
    )
).numpy()

RF_results = {
    'spatial': test_l_spatial, 
    'temporal': test_l_temporal, 
    'spatio-temporal': test_l_spatemp
}
    
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
    'Loss on spatial test data:         {}'.format(
        test_l_spatial
    )
)
print(
    'Loss on temporal test data:        {}'.format(
        test_l_temporal
    )
)
print(
    'Loss on spatio-temporal test data: {}'.format(
        test_l_spatemp
    )
)


###
# Visualize predictions and feature importance ###
###

# call feature importance plotting function
prediction.plot_RF_feature_importance(
    HYPER, 
    raw_data, 
    RF_regr.feature_importances_, 
    training_data
)

# call prediction comparison plotting function
prediction.plot_true_vs_prediction(
    'Predictions on training data with random forest', 
    training_data.Y, 
    train_pred
)
prediction.plot_true_vs_prediction(
    'Predictions on valdation data with random forest', 
    validation_data.Y, 
    val_pred
)
prediction.plot_true_vs_prediction(
    'Predictions on spatial test data with random forest', 
    spatial_test_data.Y, 
    spatial_test_pred
)
prediction.plot_true_vs_prediction(
    'Predictions on temporal test data with random forest', 
    temporal_test_data.Y, 
    temporal_test_pred
)
prediction.plot_true_vs_prediction(
    'Predictions on spatio-temporal test data with random forest', 
    spatemp_test_data.Y, 
    spatemp_test_pred
)

# delete the model as it occupies a large amount of memory
del RF_regr
_ = gc.collect()
```

    The out-of-bag validation score for random forest is: 0.32988160005843664
    Loss on training data:             0.6483886241912842
    Loss on validation data:           14.796612739562988
    Loss on spatial test data:         18.907651901245117
    Loss on temporal test data:        14.219151496887207
    Loss on spatio-temporal test data: 20.96451759338379


    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,
    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,
    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,
    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,
    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,



    
![png](images/markdown/output_23_2.png)
    



    
![png](images/markdown/output_23_3.png)
    



    
![png](images/markdown/output_23_4.png)
    



    
![png](images/markdown/output_23_5.png)
    



    
![png](images/markdown/output_23_6.png)
    



    
![png](images/markdown/output_23_7.png)
    


### 2.2 Definition

Here, we build our embedding network prediction model with an architecture that can be set through a number of hyper parameters. The model summary below contains the tensor shapes of each layer of our model, and the number of its trainable weights/parameters. The subsequent computational graphs then show the constellation of each module of our network, which we will refer to as feature encoders. The last graph shows the entire prediction model built from all these modules. 


```python
models = prediction.build_prediction_model(
    HYPER, 
    raw_data, 
    training_data
)
```

    Building prediction model
    Model: "model_4"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    X_t (InputLayer)                [(None, 4)]          0                                            
    __________________________________________________________________________________________________
    X_s1 (InputLayer)               [(None, 100, 3)]     0                                            
    __________________________________________________________________________________________________
    X_st (InputLayer)               [(None, 24, 9)]      0                                            
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 1000)         5000        X_t[0][0]                        
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 100, 1000)    4000        X_s1[0][0]                       
    __________________________________________________________________________________________________
    conv1d (Conv1D)                 (None, 23, 16)       304         X_st[0][0]                       
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 1000)         0           dense[0][0]                      
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 100000)       0           dense_2[0][0]                    
    __________________________________________________________________________________________________
    flatten_2 (Flatten)             (None, 368)          0           conv1d[0][0]                     
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 100)          100100      flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 100)          10000100    flatten_1[0][0]                  
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 100)          36900       flatten_2[0][0]                  
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 300)          0           dense_1[0][0]                    
                                                                     dense_3[0][0]                    
                                                                     dense_4[0][0]                    
    __________________________________________________________________________________________________
    dense_5 (Dense)                 (None, 1000)         301000      concatenate[0][0]                
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 100)          100100      dense_5[0][0]                    
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 1000)         101000      dense_6[0][0]                    
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 96)           96096       dense_7[0][0]                    
    ==================================================================================================
    Total params: 10,744,600
    Trainable params: 10,744,600
    Non-trainable params: 0
    __________________________________________________________________________________________________



    None


    Computation graph for X_t_encoder:



    
![png](images/markdown/output_25_3.png)
    


    ---------------------------------------------------------------------------------------------------------



    None


    Computation graph for X_s1_encoder:



    
![png](images/markdown/output_25_7.png)
    


    ---------------------------------------------------------------------------------------------------------



    None


    Computation graph for X_st_encoder:



    
![png](images/markdown/output_25_11.png)
    


    ---------------------------------------------------------------------------------------------------------



    None


    Computation graph for X_joint_encoder:



    
![png](images/markdown/output_25_15.png)
    


    ---------------------------------------------------------------------------------------------------------



    None


    Computation graph for prediction_model:



    
![png](images/markdown/output_25_19.png)
    


    ---------------------------------------------------------------------------------------------------------


### 2.3 Training

Here, we use our training dataset to learn a functional relationship between our features on the one side and our labels on the other side. We use the functional API of the tensorflow deep learning framework to do this. The plot below illustrates the training and validation loss history (y-axis) for each epoch of training (x-axis). An epoch stands for an entire iteration over the available (randomly sequenced) training dataset.

Note: All details of the training process can be seen in the src/**hyperparameters.py** file where we set hyper parameters. 


```python
# Call training and saving functions
train_loss_hist, val_loss_hist = prediction.train_model(
    HYPER, 
    models.prediction_model, 
    training_data, 
    validation_data, 
    raw_data,
    loss_object, 
    optimizer, 
    mean_loss,
    plot=True,
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
```

    Epoch 1/30
    Training:
    5/4 [=================================] - 3s 5ms/step - loss: 5.6485
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 14.6234
    Epoch 2/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 6.1155
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 14.2390
    Epoch 3/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 11.0068
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 12.4536
    Epoch 4/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 11.4880
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 13.2087
    Epoch 5/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 6.7037
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 15.9752
    Epoch 6/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 2.8936
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 16.4114
    Epoch 7/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 1.6061
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 17.8559
    Epoch 8/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 2.5192
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 12.8657
    Epoch 9/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 4.3617
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 35.9335
    Epoch 10/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 2.3585
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 12.8997
    Epoch 11/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 4.3166
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 17.9291
    Epoch 12/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 1.4278
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 17.3500
    Epoch 13/30
    Training:
    5/4 [=================================] - 0s 5ms/step - loss: 2.6524
    Validation:
    12/11 [==============================] - 0s 2ms/step - loss: 21.3180



    
![png](images/markdown/output_27_1.png)
    


    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.


### 2.4 Testing

We calculate the prediction error of our trained model on all available datasets and plot exemplar predictons for each of these. The x-axis of each plot describes the 15-min time steps into the future from a given timestamp, and the y-axis the corresponding electric consumption of a given building at these times in kW/kWh.

Note: Again, the evaluation of the training and validation datasets merely serve for a comparision to our random forest baseline predictor and for seeing whether we are able to learn useful patterns, underfit or overfit. 


```python
prediction.test_model(
    HYPER, 
    'Training data', 
    models.prediction_model, 
    training_data, 
    raw_data,
    mean_loss, 
    loss_function, 
    silent=False, 
    plot=True
)
prediction.test_model(
    HYPER, 
    'Validation data', 
    models.prediction_model, 
    validation_data, 
    raw_data,
    mean_loss, 
    loss_function, 
    silent=False, 
    plot=True
)
prediction.test_model(
    HYPER, 
    'Spatial test data', 
    models.prediction_model, 
    spatial_test_data, 
    raw_data,
    mean_loss, 
    loss_function, 
    silent=False, 
    plot=True
)
prediction.test_model(
    HYPER, 
    'Temporal test data', 
    models.prediction_model, 
    temporal_test_data, 
    raw_data,
    mean_loss,  
    loss_function, 
    silent=False, 
    plot=True
)
_ = prediction.test_model(
    HYPER, 
    'Spatio temporal test data', 
    models.prediction_model, 
    spatemp_test_data, 
    raw_data,
    mean_loss, 
    loss_function, 
    silent=False, 
    plot=True
)
```

    Training data loss: 1.2746387
    Validation data loss: 21.026152


    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,
    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,


    Spatial test data loss: 18.899147
    Temporal test data loss: 28.41303


    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,
    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,


    Spatio temporal test data loss: 33.67621


    /DataSelectionMaps/src/prediction.py:277: UserWarning: You have mixed positional and keyword arguments, some input may be discarded.
      fontsize=16,



    
![png](images/markdown/output_29_6.png)
    



    
![png](images/markdown/output_29_7.png)
    



    
![png](images/markdown/output_29_8.png)
    



    
![png](images/markdown/output_29_9.png)
    



    
![png](images/markdown/output_29_10.png)
    


## 3. Active learning

In this section, we develop our pool-based batch active learning method. First, we encode the features of candidate data points into an embedded vector space using an embedding network that is trained on initially available data points. In this case, the initially available data is our training dataset. Second, we cluster candidate data points based on their vector distances to each other in the encoded space. Third, we calculate the distance of each data embedded candidate data point to its cluster center to express our metric of informativeness. We make queries according to different metrices using these distances. The figure bellow shows how we do this for each of our query variants. We start with creating a new class object *ActLrnResults* which bundles all the results for each of the active learning variants that we evaluate. Classes and functions that we introduce in this section are defined in **activelearning.py**.

<img src="../images/markdown/QueryVariants.png">


```python
# initialize hyper parameters for AL
HYPER.set_act_lrn_params()

dataset_list = [
    spatial_test_data,
    temporal_test_data,
    spatemp_test_data
]

# for demonstration, consider all possible query variables
query_variable_list = [
    'X_t', 
    'X_s1', 
    'X_st', 
    'X_(t,s)', 
    'Y_hat_(t,s)', 
    'Y_(t,s)'
]
```

### 3.1 Encode features

We use the encoders that were previously trained alongside our prediction model to encode features into an embedded vector space. Each feature type (time, space and space-time variant) has its own encoder. All features together further have a joint encoder that can encode the respective feature vectors of each data point into an embedded vector space of some fixed dimension. The vector space into which each feature vector is embedded is defined through the last layer (embedding layer) of each encoder that we define through the architecture of our embedding network prediction model. 


```python
# create empty lists for saving encoded features
encoded_dataset_list = []

for dataset in dataset_list:

    available_index_set = set(
        np.arange(
            dataset.n_datapoints
        )
    )
    
    encoding_list = []
    
    for query_variable in query_variable_list:

        encoding, _ = activelearning.encode_features(
            HYPER, 
            raw_data, 
            models, 
            dataset,
            available_index_set,
            query_variable, 
            silent=False
        )
        
        encoding_list.append(encoding)
        
    encoded_dataset_list.append(encoding_list)
```

    Encoding features into embedded vector spaces for X_t
    Encoding features into embedded vector spaces for X_s1
    Encoding features into embedded vector spaces for X_st
    Encoding features into embedded vector spaces for X_(t,s)
    Encoding features into embedded vector spaces for Y_hat_(t,s)
    Encoding features into embedded vector spaces for Y_(t,s)
    Encoding features into embedded vector spaces for X_t
    Encoding features into embedded vector spaces for X_s1
    Encoding features into embedded vector spaces for X_st
    Encoding features into embedded vector spaces for X_(t,s)
    Encoding features into embedded vector spaces for Y_hat_(t,s)
    Encoding features into embedded vector spaces for Y_(t,s)
    Encoding features into embedded vector spaces for X_t
    Encoding features into embedded vector spaces for X_s1
    Encoding features into embedded vector spaces for X_st
    Encoding features into embedded vector spaces for X_(t,s)
    Encoding features into embedded vector spaces for Y_hat_(t,s)
    Encoding features into embedded vector spaces for Y_(t,s)


### 3.2 Compute clusters

In each encoded vector space, we can use the distances of encoded feature vectors to each other to compute clusters in our embedded data. The method that we use for clustering data points can be set through our hyper parameters which is the K-means++ algorithm here. 


```python
# create empty lists for saving results
labels_list = []
centers_list = []

for dataset, encoded_dataset in zip(
    dataset_list, 
    encoded_dataset_list
):

    l_list = []
    c_list = []

    for encoding in encoded_dataset:
        
        # Compute total data budget
        data_budget = math.floor(
            HYPER.DATA_BUDGET_ACT_LRN * dataset.n_datapoints
        )

        # compute the batch siz of this iteration
        cand_batch_size = int(
            HYPER.CAND_BATCH_SIZE_ACT_LRN * data_budget
        )
        
        # if exceeding candidate data subsample, adjust batch size
        if HYPER.CAND_SUBSAMPLE_ACT_LRN is not None:

            cand_batch_size = min(
                cand_batch_size, 
                HYPER.CAND_SUBSAMPLE_ACT_LRN
            )

        # create clusters
        labels, centers, n_clusters = activelearning.compute_clusters(
            HYPER, 
            encoding, 
            cand_batch_size, 
            silent=False
        )
        
        # add to lists
        l_list.append(labels)
        c_list.append(centers)
        
    labels_list.append(l_list)
    centers_list.append(c_list)
```

    Creating clusters in encodings with n_clusters= 35


    /DataSelectionMaps/src/activelearning.py:258: ConvergenceWarning: Number of distinct clusters (12) found smaller than n_clusters (35). Possibly due to duplicate points in X.
      clustering_method.fit(encoding)


    Creating clusters in encodings with n_clusters= 35
    Creating clusters in encodings with n_clusters= 35
    Creating clusters in encodings with n_clusters= 35
    Creating clusters in encodings with n_clusters= 35
    Creating clusters in encodings with n_clusters= 35
    Creating clusters in encodings with n_clusters= 34


    /DataSelectionMaps/src/activelearning.py:258: ConvergenceWarning: Number of distinct clusters (33) found smaller than n_clusters (34). Possibly due to duplicate points in X.
      clustering_method.fit(encoding)


    Creating clusters in encodings with n_clusters= 34


    /DataSelectionMaps/src/activelearning.py:258: ConvergenceWarning: Number of distinct clusters (32) found smaller than n_clusters (34). Possibly due to duplicate points in X.
      clustering_method.fit(encoding)


    Creating clusters in encodings with n_clusters= 34
    Creating clusters in encodings with n_clusters= 34
    Creating clusters in encodings with n_clusters= 34
    Creating clusters in encodings with n_clusters= 34
    Creating clusters in encodings with n_clusters= 81


    /DataSelectionMaps/src/activelearning.py:258: ConvergenceWarning: Number of distinct clusters (31) found smaller than n_clusters (81). Possibly due to duplicate points in X.
      clustering_method.fit(encoding)


    Creating clusters in encodings with n_clusters= 81


    /DataSelectionMaps/src/activelearning.py:258: ConvergenceWarning: Number of distinct clusters (71) found smaller than n_clusters (81). Possibly due to duplicate points in X.
      clustering_method.fit(encoding)


    Creating clusters in encodings with n_clusters= 81
    Creating clusters in encodings with n_clusters= 81
    Creating clusters in encodings with n_clusters= 81
    Creating clusters in encodings with n_clusters= 81


### 3.3 Compute distances

We can use the distance of each embedded data point (or its partial feature vectors) to its cluster center to express our metric of informativeness.


```python
for index_dataset, encoded_dataset in enumerate(encoded_dataset_list):

    l_list = labels_list[index_dataset]
    c_list = centers_list[index_dataset]

    for index_var in range(len(query_variable_list)):
        
        encoding = encoded_dataset[index_var]
        cluster_labels = l_list[index_var]
        cluster_centers = c_list[index_var]
        similarity_array =  activelearning.compute_similarity(
            HYPER, 
            encoding, 
            cluster_labels, 
            cluster_centers, 
            silent=False
        )
```

    Calculating distances
    701/701 [==============================] - 0s 239us/step
    Calculating distances
    701/701 [==============================] - 0s 210us/step
    Calculating distances
    701/701 [==============================] - 0s 226us/step
    Calculating distances
    701/701 [==============================] - 0s 224us/step
    Calculating distances
    701/701 [==============================] - 0s 220us/step
    Calculating distances
    701/701 [==============================] - 0s 213us/step
    Calculating distances
    681/681 [==============================] - 0s 211us/step
    Calculating distances
    681/681 [==============================] - 0s 213us/step
    Calculating distances
    681/681 [==============================] - 0s 212us/step
    Calculating distances
    681/681 [==============================] - 0s 208us/step
    Calculating distances
    681/681 [==============================] - 0s 210us/step
    Calculating distances
    681/681 [==============================] - 0s 210us/step
    Calculating distances
    1639/1639 [==============================] - 0s 206us/step
    Calculating distances
    1639/1639 [==============================] - 0s 207us/step
    Calculating distances
    1639/1639 [==============================] - 0s 207us/step
    Calculating distances
    1639/1639 [==============================] - 0s 202us/step
    Calculating distances
    1639/1639 [==============================] - 0s 205us/step
    Calculating distances
    1639/1639 [==============================] - 0s 207us/step


### 3.4 Batching algorithm

In each iteration of our active learning algorithm, we query a batch of candidate data points. We evaluate the performance of our algorithm for each prediction type and compare these to a passive learning benchmark. For each prediction type, we evaluate data queries according to the embedding uncertainty of our respective feature vectors. We also evaluate the performance of our algorithm for the hypothetical case of using the true labels of candidates as query variable, representing the optimal priors that we can build on candidate data. The outputs below show information about the progress of a set of relevant values of our algorithm in each iteration. Furthermore, if chosen so through hyper parameters, we conduct another set of experiments in which we randomize the sequence of actively queried data points to see whether purely the information content of queried data matters, or whether the training sequence of queried data also has an impact on training and validation losses.

Note: We plot our random forest baseline prediction losses (dashed red line) to see whether our models make reasonably good predictions or not.


```python
# initialize hyper parameters for AL
HYPER.set_act_lrn_params()

# create empty lists to add results
AL_result_list = []
PL_result_list = []

# iterate over all prediction types
for pred_type in HYPER.PRED_LIST_ACT_LRN:
    
    # choose corresponding test data of currently iterated pred_type
    if pred_type=='spatial':
        dataset = dataset_list[0]
        
    if pred_type=='temporal':
        dataset = dataset_list[1]
        
    if pred_type=='spatio-temporal':
        dataset = dataset_list[2]
        
    # create random result for benchmark once only for this pred_type
    PL_result =  activelearning.feature_embedding_AL(
        HYPER, 
        pred_type, 
        models, 
        raw_data, 
        training_data, 
        dataset, 
        loss_object, 
        optimizer, 
        mean_loss,  
        loss_function,
        'PL', 
        silent=False
    )
    
    # create empty list for saving results of corresponding AL variable
    var_result_list = []

    # iterate over all sort variables that are chosen to be considered
    for query_variable in HYPER.QUERY_VARIABLES_ACT_LRN:
    
        # empty list for savings results of correspnding AL variant
        method_result_list = []

        # iterate over all methods that are chosen to be considered
        for method in HYPER.QUERY_VARIANTS_ACT_LRN:

            # test AL with currently iterated AL variable and variant
            AL_result =  activelearning.feature_embedding_AL(
                HYPER, 
                pred_type, 
                models, 
                raw_data, 
                training_data, 
                dataset,
                loss_object, 
                optimizer, 
                mean_loss,  
                loss_function,
                method=method, 
                AL_variable=query_variable, 
                silent=False
            )
            
            # test sequence importance for iterated AL variable and variant
            AL_result = activelearning.test_AL_sequence_importance(
                HYPER, 
                pred_type, 
                models, 
                raw_data, 
                training_data, 
                dataset, 
                loss_object, 
                optimizer, 
                mean_loss,  
                loss_function, 
                AL_result,
                method=method, 
                AL_variable=query_variable, 
                silent=False
            )

            # add results to method_result_list
            method_result_list.append(AL_result)
         
         # add results to var_result_list
        var_result_list.append(method_result_list)
    
    # add results to total result_list and random_result_list
    AL_result_list.append(var_result_list)
    PL_result_list.append(PL_result)
    
# call visualization function for comparing train and validation results
activelearning.vis_train_and_val(
    HYPER, 
    AL_result_list, 
    PL_result_list, 
    RF_results
)

# call visualization function for comparing losses for random vs AL query sequence
activelearning.vis_seq_importance(
    HYPER, 
    AL_result_list
)

# save active learning results
activelearning.save_act_lrn_results(
    HYPER, 
    raw_data, 
    RF_results, 
    AL_result_list, 
    PL_result_list
)

# save hyper parameters
activelearning.save_hyper_params(
    HYPER, 
    raw_data
)

# save the prediction models
activelearning.save_act_lrn_models(
    HYPER, 
    raw_data, 
    AL_result_list, 
    PL_result_list
)

# save the test data sample
activelearning.save_act_lrn_test_sample(
    HYPER, 
    raw_data, 
    AL_result_list, 
    PL_result_list
)
```

    prediction task:             spatio-temporal
    AL variable:                 None
    AL variant:                  PL
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 24/26 (92%)
    Used sensors:                         47/65 (72%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 3s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 3s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         63/65 (97%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 3s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 3s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 3s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 2s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 3s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 2s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_st
    AL variant:                  rnd d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         38/65 (58%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         55/65 (85%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 4s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_st
    query variant:                        rnd d_c
    10/10 [==============================] - 26s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_st
    AL variant:                  min d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 8s
    Trained on candidate batch size:      81
    Used streaming times:                 24/26 (92%)
    Used sensors:                         34/65 (52%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 8s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         48/65 (74%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         56/65 (86%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_st
    query variant:                        min d_c
    10/10 [==============================] - 30s 3s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_st
    AL variant:                  max d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 8s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         24/65 (37%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         35/65 (54%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         43/65 (66%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         51/65 (78%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 8s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         55/65 (85%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         59/65 (91%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_st
    query variant:                        max d_c
    10/10 [==============================] - 27s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_st
    AL variant:                  avg d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 24/26 (92%)
    Used sensors:                         25/65 (38%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 7s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         34/65 (52%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 8s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         44/65 (68%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         48/65 (74%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         56/65 (86%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         59/65 (91%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 5s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 6s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 4s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_st
    query variant:                        avg d_c
    10/10 [==============================] - 24s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_(t,s)
    AL variant:                  rnd d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         38/65 (58%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 19s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         56/65 (86%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 8s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_(t,s)
    query variant:                        rnd d_c
    10/10 [==============================] - 27s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_(t,s)
    AL variant:                  min d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 23/26 (88%)
    Used sensors:                         28/65 (43%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         49/65 (75%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         54/65 (83%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 9s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_(t,s)
    query variant:                        min d_c
    10/10 [==============================] - 25s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_(t,s)
    AL variant:                  max d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 19/26 (73%)
    Used sensors:                         16/65 (25%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         29/65 (45%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         38/65 (58%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         48/65 (74%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         52/65 (80%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         58/65 (89%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_(t,s)
    query variant:                        max d_c
    10/10 [==============================] - 26s 3s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 X_(t,s)
    AL variant:                  avg d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 22/26 (85%)
    Used sensors:                         22/65 (34%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 18s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         40/65 (62%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 18s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         55/65 (85%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         56/65 (86%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         59/65 (91%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 8s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       X_(t,s)
    query variant:                        avg d_c
    10/10 [==============================] - 23s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_hat_(t,s)
    AL variant:                  rnd d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         28/65 (43%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 23s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         39/65 (60%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         54/65 (83%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         57/65 (88%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_hat_(t,s)
    query variant:                        rnd d_c
    10/10 [==============================] - 28s 3s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_hat_(t,s)
    AL variant:                  min d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 18s
    Trained on candidate batch size:      81
    Used streaming times:                 20/26 (77%)
    Used sensors:                         38/65 (58%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         42/65 (65%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         45/65 (69%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         50/65 (77%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         57/65 (88%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         59/65 (91%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         63/65 (97%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 9s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         63/65 (97%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_hat_(t,s)
    query variant:                        min d_c
    10/10 [==============================] - 26s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_hat_(t,s)
    AL variant:                  max d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 19/26 (73%)
    Used sensors:                         21/65 (32%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 22s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         29/65 (45%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         38/65 (58%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         41/65 (63%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         49/65 (75%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         57/65 (88%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         59/65 (91%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         60/65 (92%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_hat_(t,s)
    query variant:                        max d_c
    10/10 [==============================] - 25s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_hat_(t,s)
    AL variant:                  avg d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 19/26 (73%)
    Used sensors:                         26/65 (40%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         37/65 (57%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         43/65 (66%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         49/65 (75%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         58/65 (89%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         63/65 (97%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         63/65 (97%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 9s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         64/65 (98%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_hat_(t,s)
    query variant:                        avg d_c
    10/10 [==============================] - 26s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_(t,s)
    AL variant:                  rnd d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 22s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         17/65 (26%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 24s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         31/65 (48%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         42/65 (65%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         55/65 (85%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         63/65 (97%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 9s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         65/65 (100%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_(t,s)
    query variant:                        rnd d_c
    10/10 [==============================] - 26s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_(t,s)
    AL variant:                  min d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 22s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         16/65 (25%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 22s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         30/65 (46%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 19s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         38/65 (58%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 18s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         43/65 (66%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         50/65 (77%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         53/65 (82%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 12s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         58/65 (89%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         59/65 (91%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         61/65 (94%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         63/65 (97%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_(t,s)
    query variant:                        min d_c
    10/10 [==============================] - 31s 3s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_(t,s)
    AL variant:                  max d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 20s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         14/65 (22%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 21s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         22/65 (34%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 18s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         29/65 (45%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 17s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         34/65 (52%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 16s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         44/65 (68%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         48/65 (74%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         53/65 (82%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         55/65 (85%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         57/65 (88%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 11s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         58/65 (89%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_(t,s)
    query variant:                        max d_c
    10/10 [==============================] - 24s 2s/step
    ------------------------------------------------------------
    prediction task:             spatio-temporal
    AL variable:                 Y_(t,s)
    AL variant:                  avg d_c
    distance metric:             Laplacian
    clustering method:           KMeans
    data budget:                 819/1639 (50%)
    used sensors:                27
    new sensors to place:        65
    used streaming times:        5
    new streaming times to use:  26
    ---------
    Iteration:                            1
    Time:                                 23s
    Trained on candidate batch size:      81
    Used streaming times:                 25/26 (96%)
    Used sensors:                         18/65 (28%)
    Used data budget:                     81/819 (10%)
    ---------
    Iteration:                            2
    Time:                                 22s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         26/65 (40%)
    Used data budget:                     162/819 (20%)
    ---------
    Iteration:                            3
    Time:                                 18s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         36/65 (55%)
    Used data budget:                     243/819 (30%)
    ---------
    Iteration:                            4
    Time:                                 18s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         48/65 (74%)
    Used data budget:                     324/819 (40%)
    ---------
    Iteration:                            5
    Time:                                 15s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         54/65 (83%)
    Used data budget:                     405/819 (49%)
    ---------
    Iteration:                            6
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         57/65 (88%)
    Used data budget:                     486/819 (59%)
    ---------
    Iteration:                            7
    Time:                                 13s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         60/65 (92%)
    Used data budget:                     567/819 (69%)
    ---------
    Iteration:                            8
    Time:                                 14s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     648/819 (79%)
    ---------
    Iteration:                            9
    Time:                                 10s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     729/819 (89%)
    ---------
    Iteration:                            10
    Time:                                 9s
    Trained on candidate batch size:      81
    Used streaming times:                 26/26 (100%)
    Used sensors:                         62/65 (95%)
    Used data budget:                     810/819 (99%)
    ------------------------------------------------------------
    Testing sequence importance for
    prediction type:                      spatio-temporal
    query variable:                       Y_(t,s)
    query variant:                        avg d_c
    10/10 [==============================] - 23s 2s/step
    ------------------------------------------------------------
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.



    
![png](images/markdown/output_39_1.png)
    



    
![png](images/markdown/output_39_2.png)
    



```python

```
