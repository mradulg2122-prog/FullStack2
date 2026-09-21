import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

nb = new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (perception311)",
        "language": "python",
        "name": "perception311"
    },
    "language_info": {
        "name": "python",
        "version": "3.11"
    }
}

cells = [
    new_markdown_cell("# 🚢 Titanic Survival Prediction using Artificial Neural Networks (ANN)\nIn this notebook, we build, train, and evaluate an Artificial Neural Network (ANN) using **TensorFlow/Keras** to predict passenger survival on the Titanic dataset."),
    
    new_markdown_cell("## 1. Import Libraries"),
    new_code_cell("""import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"TensorFlow Version: {tf.__version__}")
"""),

    new_markdown_cell("## 2. Load and Inspect the Dataset"),
    new_code_cell("""df = pd.read_csv('data/train (1).csv')
print("Dataset Shape:", df.shape)
df.head()
"""),

    new_code_cell("""df.info()"""),
    
    new_code_cell("""df.describe()"""),

    new_markdown_cell("## 3. Data Cleaning and Feature Engineering"),
    new_code_cell("""print("Missing values per column:")
print(df.isnull().sum())
"""),

    new_code_cell("""# Data Preprocessing & Feature Engineering
data = df.copy()

# 1. Fill missing Age with median
data['Age'] = data['Age'].fillna(data['Age'].median())

# 2. Fill missing Embarked with mode
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# 3. Create FamilySize and IsAlone features
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# 4. Drop non-predictive / high-cardinality columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(columns=drop_cols)

# 5. One-Hot Encode categorical features
data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

print("Preprocessed DataFrame preview:")
data.head()
"""),

    new_markdown_cell("## 4. Train-Test Split and Feature Scaling"),
    new_code_cell("""# Split into Features (X) and Target (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""),

    new_markdown_cell("## 5. Build the Artificial Neural Network (ANN) Architecture"),
    new_code_cell("""# Define ANN Model Architecture
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
"""),

    new_markdown_cell("## 6. Train the Model with Early Stopping"),
    new_code_cell("""early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)
"""),

    new_markdown_cell("## 7. Model Evaluation and Performance Metrics"),
    new_code_cell("""# Evaluate on Test Set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
"""),

    new_code_cell("""# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# Classification Metrics
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
"""),

    new_code_cell("""# Visualizations: Training History & Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history.history['loss'], label='Train Loss', color='royalblue')
axes[0].plot(history.history['val_loss'], label='Val Loss', color='orange')
axes[0].set_title('Model Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Did not survive (0)', 'Survived (1)'],
            yticklabels=['Did not survive (0)', 'Survived (1)'])
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
""")
]

nb.cells = cells

# Execute the notebook
print("Executing notebook with perception311 kernel...")
client = NotebookClient(nb, timeout=600, kernel_name='perception311')
client.execute()

# Save executed notebook
with open('titanic_ANN.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("titanic_ANN.ipynb executed and saved successfully!")
