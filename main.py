import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Dataset
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'marks': [45, 50, 55, 60, 70, 75, 85, 95]
}

# Create DataFrame
df = pd.DataFrame(data)

print(df)

# Features and target
X = df[['study_hours', 'attendance']]
y = df['marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy check
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)

# Example prediction
sample = pd.DataFrame({
    'study_hours': [5],
    'attendance': [80]
})

predicted_marks = model.predict(sample)

print("Predicted Marks:", predicted_marks[0])

# Visualization
plt.scatter(df['study_hours'], df['marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")

plt.savefig("graph.png")

# Save model
joblib.dump(model, 'model.pkl')

print("Model saved successfully!")