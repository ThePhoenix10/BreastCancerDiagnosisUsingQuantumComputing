import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit.primitives import BackendSampler
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_aer import Aer

data = load_breast_cancer()
features = data.data
labels = data.target

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

std_scale = StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

samples = np.append(X_train, X_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
X_train = minmax_scale.transform(X_train)
X_test = minmax_scale.transform(X_test)

def enhanced_feature_map(num_features, reps=3):
    params = ParameterVector("x", length=num_features)
    qc = QuantumCircuit(num_features)
    for _ in range(reps):
        qc.h(range(num_features))
        for i in range(num_features):
            qc.rx(params[i], i)
            qc.ry(params[i], i)
        for i in range(num_features - 1):
            qc.cx(i, i + 1)
    return qc

backend = Aer.get_backend('aer_simulator')
sampler = BackendSampler(backend)

fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=enhanced_feature_map)

pegasos_model = PegasosQSVC(quantum_kernel=kernel, C = 600, num_steps=1100)
pegasos_model.fit(X_train, y_train)

y_pred = pegasos_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
qsvc_score = pegasos_model.score(X_test, y_test)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"QSVC classification test score: {qsvc_score}")

print("Confusion Matrix:")
print(conf_matrix)
print("-" * 50)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title(f"Confusion Matrix Without Custom Quantum Circuit")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

