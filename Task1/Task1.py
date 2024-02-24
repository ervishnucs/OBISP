from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
print("Feature names:", iris.feature_names)

# Print target names
print("Target names:", iris.target_names)

# Print the shape of the data
print("Shape of data:", iris.data.shape)

# Print the first few rows of the data
print("First few rows of data:\n", iris.data[:5])

# Print the target variable
print("Target variable:\n", iris.target)

X = iris.data  # Features
y = iris.target
from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC

# Create SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier
svm_classifier.fit(X_train, y_train)
#Predicting the species on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluating the model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))
# Example usage of the trained model for prediction
new_measurements = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8], [7.3, 2.9, 6.3, 1.8]]
predicted_species = svm_classifier.predict(new_measurements)
print("Predicted species for new measurements:",predicted_species)