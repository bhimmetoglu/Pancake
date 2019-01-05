# README

PanCake is a Python package that allows users to stack [scikit-learn](https://scikit-learn.org/stable/) models over a number 
of folds and train stacker models using out-of-sample predictions of input models.

<center>
![Stacks](https://github.com/bhimmetoglu/Pancake/blob/master/docs/img/Stacks.jpeg){height=25%, width=25%}
</center>

The stacking tool provides the construction of a `stacking module` composed of in-layer (models being stacked)
and out-layer (stacker models) models. The output is a list or matrix of predictions from training of the module,
which can either be used as the final results, or fed into a different module.

## Installation

After cloning the repository, install from the directory of the package by

```
pip install .
```

## Usage

### Initiating stacker
```python
stacker = Stacker(X, y, splitter, evalMetric, family)
```
where `X` is the data matrix (numpy array), y is target vector (numpy array),
`splitter` is a scikit-learn cross-validation generator (KFold or StratifiedKFold),
`evalMetric` is the metric to be **maximized** during training, and `family` is the 
type of the problem (currently "regression" or "binary"). 

### Adding models (in-layer):

Add a scikit-learn model `modelObj` to in-layer by
```python
stacker.addModelIn(modelObj, trainable, hyperParameters)
```
If `trainable` is set to `True` then the model will be trained across folds using the 
`hyperParameters` which is a dictionary of hyper-parameter grid for the 
model (check scikit-learn's documentation for the model). If it is set
to `False` then the model is assumed fixed and is only fitted across folds.

### Adding stacker models (out-layer):

Add a scikit-learn model `modelObj` to out-layer by
```python
stacker.addModelOut(modelObj, hyperParameters)
```
Again, `hyperParameters` is a dictionary containig the grid
of hyper-parameters for the model.

### Training and Predictions:
To train the model and get predictions on the training data, use
```python
predsTrain = stacker.stackTrain(matrixOut)
```
which yields final predictions for each out-layer model as a list when
`matrixOut` is set to `False`. When it is set to `True`, predictions
for each out-model is appended as column vectors is a an array.

For predictions on the test set, use:
```python
predsTest = stacker.stackTest(X_ts, matrixOut)
```
where `X_ts` is the test data and `matrixOut` is the same as above.

### Summary, Saving and Loading:
To get a summary on CV scores, fit and training times for each in-layer and
out-layer model, use
```python
stacker.summary()
```

To save the trained stacker for later use, call
```python
saveModel(stacker, savePath)
```
To load a trained model from disk, call
```python
stacker = loadModel(savePath)
```

### Examples
Jupyter notebooks analyzing the Boston Housing data is included in the repo:
1. [Stacking linear models](https://github.com/bhimmetoglu/Pancake/blob/master/examples/BosHousing.ipynb)
2. [Stacking Random Forest and Support Vector Regressors](https://github.com/bhimmetoglu/Pancake/blob/master/examples/BosHousing2.ipynb)

## TODO

1. Multi-class classification problems
2. Parallelization at the model and/or hyper-parameter level