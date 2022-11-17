# import our data from previous assignment
import drug_Consumption_Dataset as dcd
#import labour data
import labour_dataset as lb

from sklearn.model_selection import KFold, cross_validate
from sklearn import tree, ensemble, svm, neighbors, metrics, neural_network
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import pickle
import math


classifiers = [('DT', tree.DecisionTreeClassifier), ('RF', ensemble.RandomForestClassifier), ('SVM', svm.SVC), ('KNN' ,neighbors.KNeighborsClassifier), ('MLP', neural_network.MLPClassifier), ('GB', ensemble.GradientBoostingClassifier)]

cross_validation = KFold(n_splits=10, random_state=1, shuffle=True)



########## Functions for Cross Validate (not used since Grid Search does Kfold Cross Validation for us)
scoring = {'acc': 'accuracy',
           'prec': 'precision',
           'rec': 'recall'}

def initializeClassifier(classifier):
	if(classifier[0]=='MLP'):
		return classifier[1]()
	else:
		return classifier[1]()


def CrossValidate(classifier, features, labels, sample_balancer=None):

	if(sample_balancer==None):
		class_balancing_pipeline = initializeClassifier(classifier)
	else:
		# create pipeline in order to balance the data at every fold
		class_balancing_pipeline = make_pipeline(sample_balancer, initializeClassifier(classifier))
	
	# calculate our scores for 10 folds
	scores = cross_validate(class_balancing_pipeline, features, labels, scoring=scoring, cv=cross_validation, n_jobs=-1)
	return scores

########## Function to perform grid search and cross validation
def gridSearch(classifier, features, labels, sample_balancer=None):

	# parameter tuning dictionary for grid search
	parameters = {
	'DT' :{ 
	'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss'],
	'decisiontreeclassifier__splitter': ['best', 'random'],
	'decisiontreeclassifier__max_features': ['auto', 'sqrt', 'log2'],
	'decisiontreeclassifier__class_weight': ['balanced', None]
	},
	'RF' :{
	'randomforestclassifier__criterion': ['gini', 'entropy', 'log_loss'],
	'randomforestclassifier__class_weight': ['balanced', None]
	},
	'SVM' :{
	'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
	},
	'KNN' :{
	'kneighborsclassifier__weights': ['uniform', 'distance']
	},
	'MLP':{
	'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam']
	},
	'GB': {
	'gradientboostingclassifier__loss': ['log_loss', 'deviance', 'exponential']
	}
	}

	# create pipeline in order to balance the data at every fold
	if(sample_balancer==None):
		class_balancing_pipeline = make_pipeline(classifier[1]())

	else:
		class_balancing_pipeline = make_pipeline(sample_balancer, classifier[1]())

	# perform grid search and cross validation
	grid_search = GridSearchCV(class_balancing_pipeline, param_grid = parameters[classifier[0]], cv=cross_validation, scoring='accuracy')
	results = grid_search.fit(features, labels)
	best_model = results.best_score_
	return best_model

########## Computation for the algorithms

# initiate cross validation on the drug dataset
def scoresForCrossValidation_Drugs():
	scores_dict = {}
	i = 0
	for classifier in classifiers:
		drugs_dict = {}
		for individual_drug in dcd.drugs_subset:

			#cross validation on normal data
			scores_normal = gridSearch(classifier, dcd.respondents, dcd.drugs[individual_drug].astype('int'))

			# cross validation on oversampled data
			oversampler = RandomOverSampler(sampling_strategy='minority')
			scores_over = gridSearch(classifier, dcd.respondents, dcd.drugs[individual_drug].astype('int'), sample_balancer=oversampler)

			# cross validation on undersampled data
			undersampler = RandomUnderSampler(sampling_strategy='majority')
			scores_under = gridSearch(classifier, dcd.respondents, dcd.drugs[individual_drug].astype('int'), sample_balancer=undersampler)

			drugs_dict[individual_drug] = {'normal': scores_normal, 'over': scores_over, 'under': scores_under}
		
		scores_dict[classifier[0]]= drugs_dict

	with open('./pickles/drugs_scores', 'wb') as handle:
		pickle.dump(scores_dict, handle)

# initiate cross validation on the labour or heart dataset 
def scoresForCrossValidation_Labour_Heart(df, label_column, pickle_name):
	labels = df.pop(label_column)
	scores_dict={}
	for classifier in classifiers:

		#cross validation on normal data
		scores_normal = gridSearch(classifier, df, labels)

		# cross validation on oversampled data
		oversampler = RandomOverSampler(sampling_strategy='minority')
		scores_over = gridSearch(classifier, df, labels, sample_balancer=oversampler)

		# cross validation on undersampled data
		undersampler = RandomUnderSampler(sampling_strategy='majority')
		scores_under = gridSearch(classifier, df, labels, sample_balancer=undersampler)

		scores_dict[classifier[0]] = {'normal': scores_normal, 'over': scores_over, 'under': scores_under}

	with open('./pickles/'+pickle_name, 'wb') as handle:
		pickle.dump(scores_dict, handle)

########## print ranking results

def rankAverageMetric_drugs(scores_dict):
	for individual_drug in dcd.drugs_subset:
		print("drug chosen:", individual_drug)
		items = []
		for classifier in classifiers:
			for data in ['normal', 'over', 'under']:
				items.append((classifier[0], data, scores_dict[classifier[0]][individual_drug][data]))
		ranks = sorted(items, key=lambda x: x[2], reverse=True)
		
		for i in range(len(ranks)):

			print(i+1, ranks[i])


def rankAverageMetric_Labour_Heart(scores_dict):
	items = []
	for classifier in classifiers:
		for data in ['normal', 'over', 'under']:
			items.append((classifier[0], data, scores_dict[classifier[0]][data]))
	ranks = sorted(items, key=lambda x: x[2], reverse=True)
	
	for i in range(len(ranks)):

		print(i+1, ranks[i])


########## feature transformation functions

# feature transformation function for the labour dataset
def FeatureTransformation_Labour():
	# One hot Encode Labour relations dataframe
	labour_events_transformed = pd.get_dummies(lb.labour_events, columns=['cola', 'pension', 'vacation', 'dntl_ins', 'empl_hplan'])

	# normalise numerical columns
	for column in ['dur', 'wage1', 'wage2', 'wage3', 'hours', 'stby_pay', 'shift_diff', 'holidays']:
		labour_events_transformed[column] = MinMaxScaler().fit_transform(np.array(labour_events_transformed[column]).reshape(-1,1))

	# transform boolean columns to binary values
	for column in ['educ_allw', 'lngtrm_disabil', 'bereavement']:
		labour_events_transformed[column] = labour_events_transformed[column].map({'true': 1, 'false':0})
	labour_events_transformed['label'] = labour_events_transformed['label'].map({'good': 1, 'bad':0})

	# impute null values with 0 value
	labour_events_transformed[labour_events_transformed.isnull()] = 0

	# feature selection using variance threshold
	threshold = 0.05     
	columns_to_drop = []
	for column in labour_events_transformed.columns:
		if(labour_events_transformed[column].var()<threshold):
			columns_to_drop.append(column)
	
	print(columns_to_drop)
	labour_events_transformed.drop(columns=columns_to_drop)

	return labour_events_transformed

# feature transformation function for the heart dataset
def FeatureTransformation_Heart():
	# One hot Encode Labour relations dataframe
	ha_transformed = pd.get_dummies(ha, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])

	# normalise numerical columns
	for column in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
		ha_transformed[column] = MinMaxScaler().fit_transform(np.array(ha_transformed[column]).reshape(-1,1))

	# feature selection using variance threshold
	threshold = 0.02
	columns_to_drop = []
	for column in ha_transformed.columns:
		if(ha_transformed[column].var()<threshold):
			columns_to_drop.append(column)
	
	print(columns_to_drop)
	ha_transformed.drop(columns=columns_to_drop)

	return ha_transformed


########## function to fetch our previous cross validation results

def preProcessScores():
	with open('./pickles/heart_scores', 'rb') as handle:
		heart_scores = pickle.load(handle)

	with open('./pickles/labour_scores', 'rb') as handle:
		labour_scores = pickle.load(handle)

	with open('./pickles/drugs_scores', 'rb') as handle:
		drug_scores = pickle.load(handle)

	return heart_scores, labour_scores, drug_scores

# scoresForCrossValidation_Drugs()
# labour_events = FeatureTransformation_Labour()
# scoresForCrossValidation_Labour_Heart(labour_events, 'label', 'labour_scores')


# # # # import heart attack dataset
# ha = pd.read_csv('heart_cleveland_upload.csv')

# ha_transformed = FeatureTransformation_Heart()
# scoresForCrossValidation_Labour_Heart(ha_transformed, 'condition', 'heart_scores')

print("RESULTS:")
heart_scores, labour_scores, drug_scores = preProcessScores()
print("\nDRUGS DATASET")
rankAverageMetric_drugs(drug_scores)
print("\nLABOUR DATASET")
rankAverageMetric_Labour_Heart(labour_scores)
print('\nHEART DATASET')
rankAverageMetric_Labour_Heart(heart_scores)


# function to perform friedman test and also nemenyi if difference in distributions is found
def friedman(include_balancing_for_labour_heart=False):

	heart_scores, labour_scores, drug_scores = preProcessScores()
	scores = [heart_scores, labour_scores, drug_scores]

	# create disctionary that shows average value of metric for each dataset 
	classifier_dict = {}
	for classifier in classifiers:
		classifier_data = []
		for individual_drug in dcd.drugs_subset: 
			for data in ['normal', 'over', 'under']:
				classifier_data.append(drug_scores[classifier[0]][individual_drug][data])
		if(include_balancing_for_labour_heart):
			for data in ['normal', 'over', 'under']:
				classifier_data.append(heart_scores[classifier[0]][data])
			for data in ['normal', 'over', 'under']:	
				classifier_data.append(labour_scores[classifier[0]][data])
		else:
			
			classifier_data.append(heart_scores[classifier[0]]['normal'])
			classifier_data.append(labour_scores[classifier[0]]['normal'])

		classifier_dict[classifier[0]] = classifier_data

	# create rankings
	classifier_dict_rankings = {}
	for classifier in classifiers:
		classifier_dict_rankings[classifier[0]] = []

	for i in range(len(classifier_data)):
		row = []
		for classifier in classifiers:
			row.append((classifier[0], classifier_dict[classifier[0]][i]))
	
		row_ranking = sorted(row, key=lambda x: x[1], reverse=True)
		
		rank = 1
		for item in row_ranking:
			classifier_dict_rankings[item[0]].append(rank)
			rank+=1

	average_ranks = {}
	for classifier in classifiers:
		average_ranks[classifier[0]] = sum(classifier_dict_rankings[classifier[0]])/len(classifier_dict_rankings[classifier[0]])


	# calculate friedman and nemenyi
	stat, p = friedmanchisquare(*list(classifier_dict.values()))
	print("\n\nFRIEDMAN TEST:")
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	alpha = 0.05
	if p > alpha:
		print('FAIL TO REJECT H0: Same distributions')
	else:
		print('REJECT H0: Different distributions')
		#print('\nResults of Nemenyi Post-Hoc')

		q_value = 4.030/math.sqrt(2)
		n = len(classifier_data)
		k = len(classifiers)
		print('n =', n, 'k =', k)
		cd = q_value * math.sqrt((k*(k+1))/(6*n))
		#print(posthoc_nemenyi_friedman(np.array(list(classifier_dict.values())).T))

		
		different_algos = []
		for classifier in classifiers:
			for classifier2 in classifiers:
				if(classifier!=classifier2):
					difference = abs(average_ranks[classifier[0]]-average_ranks[classifier2[0]])
					if(difference>cd and ((classifier2[0], classifier[0], difference) not in different_algos)):
						different_algos.append((classifier[0],classifier2[0], difference))
		
		nemenyi_df = pd.DataFrame(columns=[i[0] for i in classifiers], index=[i[0] for i in classifiers])
		x=0
		for classifier in classifiers:
			row = []
			for classifier2 in classifiers:
				difference = abs(average_ranks[classifier[0]]-average_ranks[classifier2[0]])
				if(difference>cd):
					row.append("1")
				else:
					row.append("0")
			nemenyi_df[classifier[0]]= row
			x+=1

		print("The critical difference value is:", cd)
		print("\nTable of critical difference between the algorithms (1 for critical difference, 0 for no critical difference)\n")
		print(nemenyi_df)

		print('\nThe critical difference is between these algorithms:')
		for combo in different_algos:
			print('\n',combo[0], 'and', combo[1], "with a difference of", combo[2])


friedman()
