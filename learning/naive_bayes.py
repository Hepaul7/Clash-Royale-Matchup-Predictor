from sklearn.naive_bayes import GaussianNB
from matrix_loader import *

# Assuming you have the following data:
# X_train: Training deck features (shape: [num_samples, num_features])
# y_train: Training deck labels (shape: [num_samples])
# X_test: Test deck features (shape: [num_samples, num_features])

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

matrix = load_matrix()
train, validation, test = split_data(matrix)

# convert the data to torch Tensors
train = ndarray_tensor(train)
validation = ndarray_tensor(validation)
test = ndarray_tensor(test)

# split labels
x_train = train[:, :-1]
y_train = train[:, -1]
# Train the classifier
clf.fit(x_train, y_train)

x_test = test[:, :-1]
y_test = test[:, -1]
# Predict the winner for test decks
predictions = clf.predict(x_test)
print(predictions)


# # Evaluate the classifier
accuracy = clf.score(x_test, y_test)
