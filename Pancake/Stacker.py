# Python 3
# Model stacking
import time
import copy
import warnings
import pickle
import numpy as np 
import pandas as pd 
from joblib import dump, load
from collections import defaultdict
from sklearn.model_selection import ParameterGrid

"""
Package Description:
----------------------------------------------------------------------

The stacker implements calculation of predictions through a module
which contains an in-layer for initial models (to be stacked) and an
out-layer which combines predictions from the in-layer (stacker).

The in-layer passes out-of-sample (OOS) and hold-out (HO) predictions
to the out-layer, which trains and fits a second level of models.

Connections between all in and out layer models are present.
Each out-layer model returns a prediction vector. The user can 
choose the best out-layer predictions to proceed (based on the
CV score) to predicting on the test set.The other option would be 
to concatenate the predictions from all out-layer models into a 
new data matrix to be fed into another stacking module.

Example Usage:
----------------------------------------------------------------------
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from from Pancake.Stacker import Stacker

# Dataset
X, y = make_classification(n_samples=1000)
X_tr, X_ts, y_tr, y_ts = train_test_split(X,y,stratify=y, test_size=0.30)

# Splitter and stacker
splitter = KFold(n_splits=5)
stacker = Stacker(X_tr, y_tr, splitter, roc_auc_score, family="binary")

# Add 2 in-layer models
stacker.addModelIn(LogisticRegression())
stacker.addModelIn(SVC())

# Add 1 out-laye model
grid = {'C':np.logspace(-2,2,100), grid)

# Train
predsTrain = stacker.stackTrain()

# Test set
predsTest = stacker.stackTest(X_ts)

# Summary
stacker.summary()

"""

# -- Input Node of stacking of a single model
class InNode(object):
	def __init__ (self, splits, modelIn, family, trainable=False, hyperParameters=None):
		"""
		The class InNode represent the first layer models

		Input
		--------------------------------------------------------------------------------
		splits          : List of folds (HO)
		modelIn         : Model object
		family          : regression or binary
		trainable       : Is the model trainable or just to be fitted
		hyperParameters : Hyper-parameters for the trainable model

		"""
		self.splits = splits
		self.modelIn = modelIn
		self.family = family
		self.trainable = trainable
		self.hyperParameters = hyperParameters

	# Fit and return predictions for a single model on [folds] - [HO fold]
	def fitPredOOS_HO(self, X, y):
		"""
		Generate out-of-sample predictions with hold-out folds

		Input
		---------------------------------------------
		X          : Data matrix
		y          : Target vector

		Output
		---------------------------------------------
		yHatOOS    : Out-of-sample predictions 
		yHatHO     : Hold-out sample predictions
		"""
	
		# Initiate dictionaries for OOS and HO folds
		yHatOOS = {}
		yHatHO = {}

		# Loop over folds
		for ho, idx_ho in self.splits.items():

			# Splits for OOS fits
			internalSplits = {i:x for i,x in self.splits.items() if i != ho}

			# OOS - HO predictions (there will be zeros in HO indices)
			yOOS = np.zeros(len(y))

			# Loop over internal splits and fit
			for vld, idx_vld in internalSplits.items():
				# Train data to fit on
				idx_tr_ = [idx_ for f, idx_ in internalSplits.items() if f != vld]
				idx_tr = np.concatenate(idx_tr_)

				# Predict on the rest
				self.modelIn.fit(X[idx_tr], y[idx_tr])
				if self.family == 'regression':
					y_prob = self.modelIn.predict(X[idx_vld])
				else:
					y_prob = self.modelIn.predict_proba(X[idx_vld])[:,1]

				yOOS[idx_vld] = y_prob

			# Save in dictionary
			yHatOOS[ho] = yOOS

			# Also predict on HO using the rest
			idx_in_ = [idx_ for f, idx_ in self.splits.items() if f != ho]
			idx_in = np.concatenate(idx_in_)

			# Fit and predict
			self.modelIn.fit(X[idx_in], y[idx_in])
			if self.family == 'regression':
				y_prob_ho = self.modelIn.predict(X[idx_ho])
			else:
				y_prob_ho = self.modelIn.predict_proba(X[idx_ho])[:,1]

			yHatHO[ho] = y_prob_ho

		return yHatOOS, yHatHO

	# Train/Fit and return predictions for a single model on [folds] - [HO fold]
	def trainPredOOS_HO(self, X, y, evalMetric):
		"""
		Generate out-of-sample predictions with hold-out folds
		while training the in-layer model

		Input
		---------------------------------------------
		X               : Data matrix
		y               : Target vector
		hyperParameters : Hyperparameters for model
		evalMetric      : Performance metric to maximize during training

		Output
		---------------------------------------------
		yHatOOS    : Out-of-sample predictions 
		yHatHO     : Hold-out sample predictions
		"""

		# Initiate dictionaries for OOS and HO folds
		yHatOOS = {}
		yHatHO = {}

		# List of hyper-parameters
		hyper_grid = list(ParameterGrid(self.hyperParameters))

		# Initiate dict for trained model hyper-params over folds
		self.trainedHyperParams = {}

		# Loop over folds
		avg_cv_scores = []
		for ho, idx_ho in self.splits.items():

			# Splits for OOS fits
			internalSplits = {i:x for i,x in self.splits.items() if i != ho}

			# Loop over hyper-parameters (Can this be parallelized?)
			scores = []                                   # Avg CV scores for each hyper-param
			y_oos_ = [np.zeros(len(y)) for _ in range(len(hyper_grid))] # OOS predictions for all hyper-params
			for ih, dict_hyp in enumerate(hyper_grid):
				# Set model hyper-parameters
				self.modelIn.set_params(**dict_hyp)

				# Loop over internal splits
				scores_ = [] 
				for vld, idx_vld in internalSplits.items():
					# Train data
					idx_tr_ = [idx_ for f, idx_ in internalSplits.items() if f != vld]
					idx_tr = np.concatenate(idx_tr_)

					# Predict on validation set
					self.modelIn.fit(X[idx_tr], y[idx_tr])
					if self.family == 'regression':
						y_prob = self.modelIn.predict(X[idx_vld])
					else:
						y_prob = self.modelIn.predict_proba(X[idx_vld])[:,1]

					# Store the current prediction
					y_oos_[ih][idx_vld] = y_prob

					# Score
					scores_.append(evalMetric(y[idx_vld], y_prob))

				# Get mean of scores_
				scores.append(np.mean(scores_))

			# Find Best hyper-parameter and determine yOOS
			best_idx = np.argmax(scores)
			avg_cv_scores.append(np.max(scores))

			# Save best hyper-parameters
			self.trainedHyperParams[ho] = hyper_grid[best_idx]

			# Save OOS predictions in dictionary
			yHatOOS[ho] = y_oos_[best_idx]

			# Also predict on HO using the rest
			idx_in_ = [idx_ for f, idx_ in self.splits.items() if f != ho]
			idx_in = np.concatenate(idx_in_)

			# Fit and predict
			self.modelIn.set_params(**hyper_grid[best_idx])
			self.modelIn.fit(X[idx_in], y[idx_in])
			if self.family == 'regression':
				y_prob_ho = self.modelIn.predict(X[idx_ho])
			else:
				y_prob_ho = self.modelIn.predict_proba(X[idx_ho])[:,1]

			yHatHO[ho] = y_prob_ho

		return yHatOOS, yHatHO, avg_cv_scores

	def fitPredOOS(self, X, y):
		"""
		Generate out-of-sample(OOS) predictions WITHOUT hold-out(HO) folds

		Input
		-----------------------------------------------------------------------
		X          : Data matrix
		y          : Target vector

		Output
		-----------------------------------------------------------------------
		yHatOOS            : Out-of-sample predictions
		dict_fittedModels  : Dictionary of fitted models for each fold
		"""
	
		# Initiate OOS predictions
		yHatOOS = np.zeros(len(y))

		# Initiate dict for fitted models over folds
		dict_fittedModels = {}

		# Loop over folds
		for vld, idx_vld in self.splits.items():
			# Indices of samples to fit on
			idx_tr_ = [idx_ for f, idx_ in self.splits.items() if f != vld]
			idx_tr = np.concatenate(idx_tr_)

			# Fit and predict
			if self.trainable:
				self.modelIn.set_params(**self.trainedHyperParams[vld])

			self.modelIn.fit(X[idx_tr], y[idx_tr])
			if self.family == 'regression':
				y_prob = self.modelIn.predict(X[idx_vld])
			else:
				y_prob = self.modelIn.predict_proba(X[idx_vld])[:,1]

			yHatOOS[idx_vld] = y_prob

			# Save model
			model_copy = copy.deepcopy(self.modelIn) # A new copy of model, not a reference to it
			dict_fittedModels[vld] = model_copy

		return yHatOOS, dict_fittedModels

	def predOOS_test(self, X_ts, dict_fittedModels, testSplits):
		"""
		Predictions on an independent test set

		Input
		-----------------------------------------------------------------------
		X_ts              : Data matrix (test set)
		dict_fittedModels : Dictionary of fitted models for each fold
		testSplits        : List of folds for test set. Number of folds must be
		                    same with dict_fittedModels. Recommended: Use the 
		                    same random_state as well.

		Output
		-----------------------------------------------------------------------
		yHatOOS    : Out-of-sample predictions (test set)
		"""

		# Initiate OOS predictions (folds x folds combinations)
		yHatOOS = np.zeros((len(X_ts),len(dict_fittedModels)))

		# Loop over folds and just predict
		for vld, idx_vld in testSplits.items():

			# Predict with all models
			for m,mod in dict_fittedModels.items():
				if self.family == 'regression':
					y_prob = mod.predict(X_ts[idx_vld])
				else:
					y_prob = mod.predict_proba(X_ts[idx_vld])[:,1]

				yHatOOS[idx_vld,m] = y_prob

		# Return predictions
		return yHatOOS.mean(axis=1)

# -- Node in out level
class OutNode(object):
	def __init__ (self, splits, modelOut, hyperParameters, evalMetric, family):
		"""
		The class OutNode represent the second layer models
		
		Input
		---------------------------------------------------------------------
		splits             : List of folds (HO)
		modelOut           : Model object
		hyperParameters    : Hyperparameters  for model
		evalMetric         : Performance metric to maximize during training
		family             : regression or binary (must match first layer)
		"""

		self.splits = splits
		self.modelOut = modelOut
		self.hyperParameters = hyperParameters
		self.evalMetric = evalMetric
		self.family = family

	def train(self, y, dict_Xoos, dict_Xho):
		"""
		Train modelOut

		Input
		--------------------------------------------------------------
		y             : Target vector
		dict_Xoos     : OOS predictions for each model in in-layer
		dict_Xho      : HO predictions for each model in in-layer

		Effect/Output
		--------------------------------------------------------------
		best_hyperParams : Best set of hyper-parameters
		best_cvscore     : CV score for the best parameters
		"""

		# List of hyper-parameters
		hyper_grid = list(ParameterGrid(self.hyperParameters))

		# Initiate dict for saving CV results
		hyper_evals = defaultdict(list)
	
		# Loop over folds
		for ho, idx_ho in self.splits.items():
			# Construct the train (X_oos) and test (X_ho) data for CV
			# For multiclass: vstack --> hstack and no T
			X_oos = np.vstack(dict_Xoos[ho]).T
			X_ho = np.vstack(dict_Xho[ho]).T

			# Remove zeros in OOS from hold-out section
			X_oos = np.delete(X_oos, idx_ho, axis=0)
			y_oos = np.delete(y, idx_ho, axis=0)
		
			# Hold-out target
			y_ho = y[idx_ho]

			# Scores on folds (can parallelize this by joblib)
			for ih, dict_hyp in enumerate(hyper_grid):
				self.modelOut.set_params(**dict_hyp)
				self.modelOut.fit(X_oos, y_oos)

				if self.family == 'regression':
					y_prob = self.modelOut.predict(X_ho)
				else:
					y_prob = self.modelOut.predict_proba(X_ho)[:,1]

				hyper_evals[ih].append(self.evalMetric(y_ho, y_prob))

		# Get best hyper-parameters with best performance
		cv_scores = [np.mean(ls_) for i, ls_ in hyper_evals.items()]
		best_idx = np.argmax(cv_scores)

		self.best_hyperParams = hyper_grid[best_idx]

		# Return best hyper-parameters and the corresponding CV score
		return self.best_hyperParams, cv_scores[best_idx]

	def fitPred(self, y, list_Xoos):
		"""
		Fit model on all training folds with best hyper-parameters

		Input
		---------------------------------------------------------------
		y                : Target vector
		list_Xoos        : OOS predictions from in-layer

		Output
		---------------------------------------------------------------
		y_final          : Final prediction 

		"""
		Xoos = np.vstack(list_Xoos).T
		self.modelOut.set_params(**self.best_hyperParams)
		self.modelOut.fit(Xoos,y)

		if self.family == 'regression':
			y_final = self.modelOut.predict(Xoos)
		else:
			y_final = self.modelOut.predict_proba(Xoos)[:,1]

		return y_final

	def predTest(self, list_Xoos):
		"""
		Predict on all test folds with best hyper-parameters

		Input
		-----------------------------------------------------------------
		list_Xoos   : OOS predictions from in-level models (test set)

		Output
		-----------------------------------------------------------------
		y_final_ts   : Final prediction (test set)

		"""

		# Just predict
		Xoos = np.vstack(list_Xoos).T
		if self.family == 'regression':
			y_final_ts = self.modelOut.predict(Xoos)
		else:
			y_final_ts = self.modelOut.predict_proba(Xoos)[:,1]

		return y_final_ts

# -- An object for generating the stacked network
class Stacker(object):
	def __init__ (self, X, y, splitter, evalMetric, family = "regression"):
		"""
		The Stacker class implements the full stacked predictions pipeline
		
		Input
		---------------------------------------------------------------------
		X             : Original data matrix
		y             : Target vector
		splitter      : Cross-validation generator
		evalMetric    : Performance metric to maximize during training
		family        : regression or binary (must match in-layer)
		"""
		self.X = X
		self.y = y
		self.splitter = splitter
		self.evalMetric = evalMetric
		self.family = family

		# Initiate lists for models in layers
		self.modelsIn = []
		self.modelsOut = []

		# Splits for training
		splt_ = list(splitter.split(X,y))
		self.splits = {i:x[1] for i,x in enumerate(splt_)}

	def addModelIn(self, modelObj, trainable = False, hyperParameters = None):
		""" Adder of model to in-layer """
		newNode = InNode(self.splits, modelObj, self.family, trainable, hyperParameters)
		self.modelsIn.append(newNode)

	def addModelOut(self, modelObj, hyperParameters): 
		""" Adder of model to out-layer """
		newNode = OutNode(self.splits, modelObj, hyperParameters, self.evalMetric, self.family)
		current_index = len(self.modelsOut)
		self.modelsOut.append(newNode)

	def _checkInLayer(self):
		""" Performs checks on the model """

		# Classification problem
		if self.family != 'regression' and len(set(self.y)) > 2:
			raise NotImplementedError("Multi-class case not implemented")

		# Number of models in the in-layer
		nIn = len(self.modelsIn)
		if nIn <= 1:
			warnings.warn("Stacking requires more then one model",RuntimeWarning)

		return 1

	def _checkOutLayer(self):
		""" Performs checks on the model """
		
		# Number of models in the out-layer
		nOut = len(self.modelsOut)
		if nOut < 1:
			warnings.warn("Stacking requires at least one out-layer model", RuntimeWarning)

		# Hyper-parameters for each model
		for m in self.modelsOut:
			if len(m.hyperParameters) < 1:
				warnings.warn("Hyperparameters need to be specified for training", RuntimeWarning)

		return 1

	def _stackLevelInTrain(self):
		""" Runs in-layer and obtain OOS & HO predictions """

		# Check in-layer model
		_ = self._checkInLayer()
		
		# The lists contain predictions from each model
		dict_Xoos = defaultdict(list)
		dict_Xho = defaultdict(list)
		
		# For each model (node) in-layer, perform predictions
		for i, node in enumerate(self.modelsIn):
			# OOS and HO predictions
			if not node.trainable:
				yHatOOS, yHatHO = node.fitPredOOS_HO(self.X, self.y)
			else:
				yHatOOS, yHatHO, avg_scores = node.trainPredOOS_HO(self.X, self.y, self.evalMetric)

			for ho in self.splits:
				dict_Xoos[ho].append(yHatOOS[ho])
				dict_Xho[ho].append(yHatHO[ho])

			# Print report 
			if node.trainable:
				print("In-layer model : {:d} trained, Avg CV score across HO folds: {:.4f}".format(i+1, np.mean(avg_scores)))
			else:
				print("In-layer model : {:d} only fitted".format(i+1))

		return dict_Xoos, dict_Xho

	def _stackLevelInFitPred(self):
		""" Runs in-layer and obtain OOS predictions on all folds """
		list_Xoos = []
	
		# List of fitted models (save in object to re-use)
		self.list_fittedModelsIn = []		
		
		# For each model (node) in-layer, perform predictions
		for node in self.modelsIn:
			yHatOOS, dict_fittedModels = node.fitPredOOS(self.X, self.y)
			list_Xoos.append(yHatOOS)
			self.list_fittedModelsIn.append(dict_fittedModels)

		return list_Xoos

	def _stackLevelInPredTest(self, X_ts, testSplits):
		""" Runs in-layer and obtain OOS predictions on test set """
		list_Xoos = []

		# Loop over models and get predictions
		for i, node in enumerate(self.modelsIn):
			yHatOOS = node.predOOS_test(X_ts, self.list_fittedModelsIn[i], testSplits)
			list_Xoos.append(yHatOOS)

		return list_Xoos

	def _trainOutLayer(self, dict_Xoos, dict_Xho):
		""" Trains out-layer to get best hyper-parameters """
		self.best_hypers = {}

		# Check out-layer
		_ = self._checkOutLayer()

		# CV scores
		self.cv_scores =[]

		for i, node in enumerate(self.modelsOut):
			best_hyper, score = node.train(self.y, dict_Xoos, dict_Xho)
			self.best_hypers[i] = best_hyper
			self.cv_scores.append(score)

			# Print report
			print("Out-layer model : {:d} trained,  CV score = {:.4f}".format(i+1, score))

	def _fitOutLayer(self, list_Xoos):
		""" Fits out-layer models on training set """

		# List of model predictions
		predsTrain = []

		for i, node in enumerate(self.modelsOut):
			y_pred = node.fitPred(self.y, list_Xoos)
			predsTrain.append(y_pred)

		return predsTrain

	def _predOutLayerTest(self, list_Xoos, testSplits):
		""" Predicts on an independent test set """
		predsTest = []

		for i, node in enumerate(self.modelsOut):
			y_prob_ts = node.predTest(list_Xoos)
			predsTest.append(y_prob_ts)

		return predsTest

	def stackTrain(self, matrixOut = False):
		""" Runner of training """

		t0 = time.time()
		# Stack in-level models for training (OOS & HO)
		dict_Xoos, dict_Xho = self._stackLevelInTrain()

		self.stackTime = time.time() - t0

		# Train out-layer for best hyper-parameters
		t0 = time.time()
		self._trainOutLayer(dict_Xoos, dict_Xho)

		# Get OOS predictions for all training set with all second level models
		list_Xoos = self._stackLevelInFitPred()
		predsTrain = self._fitOutLayer(list_Xoos)

		# Train time
		self.trainTime = time.time() - t0

		if matrixOut:
			return self._outMatrix(predsTrain)
		else:
			return predsTrain

	def stackTest(self, X_ts, matrixOut = False):
		""" Runner of test set predictions """

		t0 = time.time()

		# Get test splits
		splt_ = list(self.splitter.split(X_ts))
		testSplits = {i:x[1] for i,x in enumerate(splt_)}

		# First level OOS predictions
		list_Xoos = self._stackLevelInPredTest(X_ts, testSplits)

		# Get OOS predictions for the test set with final model
		predsTest = self._predOutLayerTest(list_Xoos, testSplits)

		# Test time
		self.testTime = time.time() - t0

		if matrixOut:
			return self._outMatrix(predsTest)
		else:
			return predsTest

	@staticmethod
	def _outMatrix(preds):
		""" Returns predictions as a matrix from a list """
		newDataMatrix = []
		for pred in preds:
			newDataMatrix.append(pred.reshape(-1,1))

		newDataMatrix = np.hstack(newDataMatrix)

		return newDataMatrix

	def summary(self):
		""" Prints summary on model training """

		# Number of models (In and Out)
		print("Stacked Model Summary:")
		print(''.join(['-']*40))
		print("{:d} in-layer models stacked in {:.2e} sec".format(len(self.modelsIn),self.stackTime))
		
		# Training CV scores 
		print("{:d} out-layer models trained in {:.2e} sec".format(len(self.modelsOut),self.trainTime))

		print("In-layer summary:")
		print(''.join(['-']*40))
		print("{:d} in-Layer models trained/fitted".format(len(self.modelsIn)))

		print("Out-layer summary:")
		print(''.join(['-']*40))
		for i,mod in enumerate(self.modelsOut):
			print("Out-layer model {:d}: CV score = {:.4f}".format(i+1,self.cv_scores[i]))
			print("Best hyper-parameters:", mod.best_hyperParams)

# -- Save and load models
def saveModel(stacker, savePath):
	""" Save trained model on disk """
	dump(stacker, savePath)

def loadModel(loadPath):
	""" Load trained model on disk """
	stk = load(loadPath)
	return stk
