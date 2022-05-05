#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import scipy
import numpy as np
import matplotlib
import pandas
import sklearn
import statsmodels
import theano
import keras

print('Python: ', sys.version)
print('scipy: ', scipy.__version__)
print('numpy: ', np.__version__)
print('matplotlib: ', matplotlib.__version__)
print('pandas: ', pandas.__version__)
print('sklearn: ', sklearn.__version__)
print('Statsmodels: ', statsmodels.__version__)
print('Theano: ', theano.__version__)
print('Keras: ', keras.__version__)


# In[2]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[3]:


url = "C:\\PhD\\Fall 2021\\Courses\\ECE 593-Introduction to Machine Learning\\Final Project\\MoNbTaTiW_1200\\updated_recordfile_1200.csv"
names = ["% Cr", "% Hf", "% Mo", "%Nb", "%Ta", "%Ti", "%Re", "%V", "%W", "%Zr", "%Co", "%Ni", "%Fe", "%Al", "%Mn", "%Cu", "%C", "Converted Entropy", "Bulk modulus", "Melting point", "VEC", "Testing T (Degree Celcius)", "Density, kgm-3", "constitution of alloy", "Yield strength (Mpa)"]
dataset = pandas.read_csv(url, names=names)
print(dataset.shape)


# In[4]:


obj_columns = dataset.select_dtypes(['object']).columns
print(obj_columns)


# In[5]:


for col_name in dataset.columns:
    if(dataset[col_name].dtype == 'object'):
        dataset[col_name]= dataset[col_name].astype('category')

cat_columns = dataset.select_dtypes(['category']).columns
print(cat_columns)


# In[6]:


dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
print(dataset.head(20))


# In[7]:


print(dataset.groupby('Yield strength (Mpa)').size())


# In[9]:


# split into input (X) and output (Y) variables
array = dataset.values
X = array[:,0:24]
Y = array[:,24]


# In[10]:


# encode class values as integers
encoder_before = LabelEncoder()
encoder_before.fit(Y)
encodedbefore_Y = encoder_before.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummybefore_y = np_utils.to_categorical(encodedbefore_Y)


# In[11]:


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=24, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, input_dim=24, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[12]:


# evaluate model with standardized dataset
validation_size = 0.10
seed = 7

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
net_kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# In[13]:


#Split-out validation dataset
net_X_train, net_X_validation, net_Y_train, net_Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# In[14]:


net_results = cross_val_score(pipeline, net_X_train, net_Y_train, cv=net_kfold)
print("Neural Network: ")
print("Mean: ", net_results.mean()*100,"Std: ", net_results.std()*100)


# In[15]:


# Predicting the Test set results
from sklearn.metrics import r2_score

pipeline.fit(net_X_train, net_Y_train)
net_predictions = pipeline.predict(net_X_validation)
print("Accuracy of Neural Network = ", r2_score(net_Y_validation, net_predictions)*100)


# In[16]:


test_url = "C:\\PhD\\Fall 2021\\Courses\\ECE 593-Introduction to Machine Learning\\Final Project\\MoNbTaTiW_1200\\test_data_1200.csv"
test_dataset = pandas.read_csv(test_url, names=names)
print(test_dataset.shape)


# In[17]:


test_obj_columns = test_dataset.select_dtypes(['object']).columns
print(test_obj_columns)


# In[18]:


for col_name in test_dataset.columns:
    if(test_dataset[col_name].dtype == 'object'):
        test_dataset[col_name]= test_dataset[col_name].astype('category')

test_cat_columns = test_dataset.select_dtypes(['category']).columns
print(test_cat_columns)


# In[19]:


test_dataset[test_cat_columns] = test_dataset[test_cat_columns].apply(lambda x: x.cat.codes)
print(test_dataset.head())


# In[20]:


# split into input (X) and output (Y) variables
test_array = test_dataset.values
test_X = test_array[:,0:24]
test_Y = test_array[:,24]


# In[21]:


test_net_predictions = pipeline.predict(test_X)
print(test_net_predictions)


# In[22]:


print(test_Y)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
np.random.seed(1)


# In[24]:


shape = np.shape(net_X_train)
print("Shape of the training dataset: ",shape)
print("Size of Training Data set before feature selection:", (net_X_train.nbytes/1e6), "MB")


# In[25]:


# random forest for feature importance on a regression problem
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, Y)
importance = model.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[26]:


#Select features which have higher contribution in the final prediction

sfm = SelectFromModel(model, threshold=0.0003) 
sfm.fit(net_X_train,net_Y_train)


# In[27]:


#Transform input dataset

net_X_train_1 = sfm.transform(net_X_train) 
net_X_validation_1 = sfm.transform(net_X_validation)


# In[28]:


#Let's see the size and shape of new dataset 

print("Size of Data set after feature selection:", (net_X_train_1.nbytes/1e6), "MB")
shape = np.shape(net_X_train_1)
print("Shape of the dataset:", shape)


# In[29]:


model.fit(net_X_train_1, net_Y_train) 

#Let's evaluate the model on test data

pre = model.predict(net_X_validation_1) 
count = 0
print("Accuracy score after feature selection:", r2_score(net_Y_validation, pre) * 100)


# In[30]:


test_X = sfm.transform(test_X)
test_pre = model.predict(test_X) 
print(test_pre)


# In[31]:


print(test_Y)


# In[32]:


print(net_Y_validation.astype(int))
print(pre)


# In[45]:


straight_x = net_Y_train
straight_y = straight_x
plt.title('Yield strength prediction of MoNbTaTiW at 1200 degree celsius')
plt.xlabel('Experimental Yield, mpa')
plt.ylabel('Prediction Yield, mpa')
plt.plot(straight_x, straight_y, '-r') #, label='Testing data'
plt.scatter(net_Y_train, net_Y_train, c='blue', label='Training data')
plt.scatter(net_Y_validation, pre, c='orange', label='Testing data')
plt.scatter(test_Y, test_pre, c='cyan')
plt.legend(loc='upper left')
plt.show()


# In[36]:


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
   # axes[0].set_xlabel("Training examples")
    #axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Experimental yield, HV")
    axes[0].set_ylabel("Prediction yield, HV")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    #axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                # label="Training score")
    #axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 #label="Cross-validation score")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training data")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Testing data")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# In[37]:


"""
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = "Learning Curves (Random Forest Regressor)"
cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
plot_learning_curve(model, title, X, Y, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()
"""


# In[38]:


# define base model
def original_baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(34, input_dim=17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(34, input_dim=17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[39]:


# evaluate model with standardized dataset

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=original_baseline_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
net_kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# In[40]:


net_results = cross_val_score(pipeline, net_X_train_1, net_Y_train, cv=net_kfold)
print("Neural Network: ")
print("Mean: ", net_results.mean()*100,"Std: ", net_results.std()*100)


# In[41]:


# Predicting the Test set results
from sklearn.metrics import r2_score

pipeline.fit(net_X_train_1, net_Y_train)
net_predictions = pipeline.predict(net_X_validation_1)
print("Accuracy of Neural Network after Feature Selection= ", r2_score(net_Y_validation, net_predictions)*100)


# In[42]:


test_net_predictions = pipeline.predict(test_X)
print(test_net_predictions)


# In[43]:


print(test_Y)


# In[ ]:

straight_x = net_Y_train
straight_y = straight_x
plt.title('Neural Network Model')
plt.xlabel('Experimental Yield, mpa')
plt.ylabel('Prediction Yield, mpa')
plt.plot(straight_x, straight_y, '-r') #, label='Testing data'
plt.scatter(net_Y_train, net_Y_train, c='blue', label='Training data')
plt.scatter(net_Y_validation, net_predictions, c='orange', label='Testing data')
plt.scatter(test_Y, test_net_predictions, c='red')
plt.legend(loc='upper left')
plt.show()



# In[ ]:




