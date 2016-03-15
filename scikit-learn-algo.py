import numpy
import pandas
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer


def encode_categorical_features(train, test):
    dict_vectorizer = DictVectorizer()
    categorical_features = []

    for column in train:
        if train[column].dtype == 'object':
            categorical_features.append(column)

    train_categorical_features = train[categorical_features].to_dict(orient='records')
    test_categorical_features = test[categorical_features].to_dict(orient='records')

    dict_vectorizer.fit(train_categorical_features)

    train_categorical_encoded = pandas.DataFrame(dict_vectorizer.transform(train_categorical_features).toarray())
    test_categorical_encoded = pandas.DataFrame(dict_vectorizer.transform(test_categorical_features).toarray())
    
    train = train.drop(categorical_features, axis=1)
    train = train.join(train_categorical_encoded)

    test = test.drop(categorical_features, axis=1)
    test = test.join(test_categorical_encoded)

    return train, test


training_file_path = "data-science-puzzle/train.csv"
test_file_path = "data-science-puzzle/test.csv"
submission_file_path = "submission.csv"

training_data = pandas.read_csv(training_file_path, header=0)
test_data = pandas.read_csv(test_file_path, header=0)

training_data, test_data = encode_categorical_features(training_data, test_data)

features = list(training_data.columns.values)
features.remove('id')
features.remove('target')

test_features = list(test_data.columns.values)
test_features.remove('id')

test_ids = test_data['id']
features_matrix = training_data.as_matrix(features)
test_matrix = test_data.as_matrix(test_features)


standard_scaler = preprocessing.StandardScaler()
features_matrix = standard_scaler.fit_transform(features_matrix)
test_matrix = standard_scaler.transform(test_matrix)
target = standard_scaler.fit_transform(training_data['target'])

# http://kukuruku.co/hub/python/introduction-to-machine-learning-with-python-andscikit-learn
# http://stats.stackexchange.com/questions/95797/how-to-split-the-dataset-for-cross-validation-learning-curve-and-final-evaluat
# http://scikit-learn.org/stable/auto_examples/




train_set, test_set, train_target_set, test_target_set = train_test_split(features_matrix, target, test_size=0.2)

# import csv
# with open("train_set_output.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(train_set)


# http://skll.readthedocs.org/en/latest/run_experiment.html

clf = svm.SVR()                                          # R2 = 0.2731
# clf = linear_model.LassoLars(alpha=.1)                   # R2 = -0.0007
# clf = linear_model.BayesianRidge()                       # R2 = 0.2463
# clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])      # R2 = 0.3043
# clf = DecisionTreeRegressor(max_depth=4)                 # R2 = 0.0035
# clf = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)   # R2 = 0.1590
# clf = RandomForestRegressor(n_estimators=50)             # R2 = 0.2707
# clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=numpy.random.RandomState(1))  # R2 = -7.513


clf.fit(train_set, train_target_set)

# test prediction
test_predictions = clf.predict(test_set)

r2_score_value = r2_score(test_target_set, test_predictions)
explained_variance_score_value = explained_variance_score(test_target_set, test_predictions)
mean_absolute_error_value = mean_absolute_error(test_target_set, test_predictions)
mean_squared_error_value = mean_squared_error(test_target_set, test_predictions)
median_absolute_error_value = median_absolute_error(test_target_set, test_predictions)

print('R2 score: ' + str(r2_score_value))
print('Mean squared error: ' + str(mean_squared_error_value))
print('Explained variance score: ' + str(explained_variance_score_value))
print('Mean absolute error: ' + str(mean_absolute_error_value))
print('Median absolute error: ' + str(median_absolute_error_value))


# prediction to submit
# predictions = clf.predict(test_matrix)
# predictions = standard_scaler.inverse_transform(predictions)
# predictions_data_frame = pandas.DataFrame({'id': test_ids, 'prediction': predictions})
# predictions_data_frame.to_csv(submission_file_path, sep=',', encoding='utf-8', index=False)
