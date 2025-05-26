import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.primitives import BackendSampler
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_aer import Aer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

data = load_breast_cancer()
features = data.data
labels = data.target

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize and scale data
std_scale = StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

samples = np.append(X_train, X_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
X_train = minmax_scale.transform(X_train)
X_test = minmax_scale.transform(X_test)

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define the enhanced quantum feature map..
def enhanced_feature_map(num_features, reps=3):
   params = ParameterVector("x", length=num_features)
   qc = QuantumCircuit(num_features)
   for _ in range(reps):
       qc.h(range(num_features))
       for i in range(num_features):
           qc.rx(params[i], i)
           qc.ry(params[i], i)
       for i in range(num_features):
           for j in range(i + 1, num_features):
               qc.cx(i, j)
   return qc

num_features = 2
feature_map = enhanced_feature_map(num_features=num_features, reps=3)

backend = Aer.get_backend('aer_simulator')
sampler = BackendSampler(backend)

fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

best_score = 0
best_params = {}
for C in [600,100]:
   for num_steps in [1100,100]:
       model = PegasosQSVC(quantum_kernel=kernel, C=C, num_steps=num_steps)
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)

    #testing  metricsclear
       accuracy = accuracy_score(y_test, y_pred)
       precision = precision_score(y_test, y_pred)
       recall = recall_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred)
       qsvc_score = model.score(X_test, y_test)
       conf_matrix = confusion_matrix(y_test, y_pred)


       train_pred = model.predict(X_train)
       train_accuracy = accuracy_score(y_train, train_pred)
       train_precision = precision_score(y_train, train_pred)
       train_recall = recall_score(y_train, train_pred)
       train_f1 = f1_score(y_train, train_pred)


    #    print(f"Hyperparameters: C={C}, num_steps={num_steps}")
       print(f"Accuracy: {accuracy}")
       print(f"Precision: {precision}")
       print(f"Recall: {recall}")
       print(f"F1 Score: {f1}")
       print(conf_matrix)
       print("-" * 50)

       
       print("\nTraining Metrics:")
       print(f"Accuracy: {train_accuracy}")
       print(f"Precision: {train_precision}")
       print(f"Recall: {train_recall}")
       print(f"F1 Score: {train_f1}")

       # Calculate ROC curve and AUC
      #  y_probs = model.decision_function(X_test)  # Get decision scores
      #  fpr, tpr, thresholds = roc_curve(y_test, y_probs)
      #  roc_auc = auc(fpr, tpr)

      #   # Plot ROC Curve
      #  plt.figure(figsize=(8, 6))
      #  plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
      #  plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")
      #  plt.xlabel('False Positive Rate')
      #  plt.ylabel('True Positive Rate')
      #  plt.title(f'ROC Curve (C={C}, num_steps={num_steps})')
      #  plt.legend(loc="lower right")
      #  plt.grid()
      #  plt.show()


    #    # Plot confusion matrix using Seaborn
    #    plt.figure(figsize=(8, 6))
    #    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    #    plt.title(f"Confusion Matrix (C={C}, num_steps={num_steps})")
    #    plt.xlabel("Predicted Labels")
    #    plt.ylabel("True Labels")
    #    plt.show()

    
           
       

       print("updating")
       # Track the best hyperparameters
       if qsvc_score > best_score:
           best_score = qsvc_score
           best_params = {"C": C, "num_steps": num_steps}

       # Print a message indicating the next iteration
       print(f"Starting next iteration...\n{'=' * 50}")


# Print the best parameters and score
print("\nBest score:", best_score)
print("Best params:", best_params)




