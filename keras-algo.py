import numpy
import pandas
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from keras.models import Sequential
from keras.layers.core import Dense, Activation

from sklearn.feature_extraction import DictVectorizer

def encode_categorical_features(train, test):
    dict_vectorizer = DictVectorizer()
    categorical_features = []

    for column in train:
        if train[column].dtype == 'object':
        	categorical_features.append(column)

    train_categorical_features = train[categorical_features].to_dict(outtype='records').toarray()
    test_categorical_features = test[categorical_features].to_dict(outtype='records').toarray()

    dict_vectorizer.fit(train_categorical_features)

	train_categorical_encoded = pandas.DataFrame(dict_vectorizer.transform(train_categorical_features))
	test_categorical_encoded = pandas.DataFrame(dict_vectorizer.transform(test_categorical_features))
    
    train = train.drop(categorical_features, axis=1)
    train = train.join(train_categorical_encoded)

    test = test.drop(categorical_features, axis=1)
    test = test.join(test_categorical_encoded)

    return train, test


training_file_path = "data-science-puzzle/train.csv"
test_file_path = "data-science-puzzle/test.csv"
submission_file_path = "submission.csv"

training_data = pandas.read_csv(training_file_path, header=0)
training_data_headers = list(training_data.columns.values)

test_data = pandas.read_csv(test_file_path, header=0)
test_data_headers = list(test_data.columns.values)

features = list(training_data_headers)
features.remove('id')
features.remove('target')

test_features = list(test_data_headers)
test_features.remove('id')

training_data, test_data = encode_categorical_features(training_data, test_data)

test_ids = test_data['id']
features_matrix = training_data.as_matrix(features)
test_matrix = test_data.as_matrix(test_features)


standard_scaler = preprocessing.StandardScaler()
features_matrix = standard_scaler.fit_transform(features_matrix)
test_matrix = standard_scaler.transform(test_matrix)
target = standard_scaler.fit_transform(training_data['target'])


train_set, test_set, train_target_set, test_target_set = train_test_split(features_matrix, target, test_size=0.2)


model = Sequential()
model.add(Dense(2, 1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

model.fit(train_set, train_target_set, nb_epoch=1000, batch_size=16, verbose=0)
model.fit(train_set, train_target_set, nb_epoch=1, batch_size=16, verbose=1)

score = model.evaluate(test_set, test_target_set, batch_size=16)

print('score: ' + str(score))


# prediction to submit
# predictions = clf.predict(test_matrix)
# predictions = standard_scaler.inverse_transform(predictions)
# predictions_data_frame = pandas.DataFrame({'id': test_ids, 'prediction': predictions})
# predictions_data_frame.to_csv(submission_file_path, sep=',', encoding='utf-8', index=False)
