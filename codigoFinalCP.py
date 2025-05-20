import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, mean_squared_error

# ================================
# Funções auxiliares da MLP
# ================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def forward(X, weights, biases):
    activations = [X]
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        a = sigmoid(z)
        activations.append(a)
    return activations

def backpropagation(y_true, activations, weights):
    grads_w = [None] * len(weights)
    grads_b = [None] * len(weights)
    error = (y_true - activations[-1]) * sigmoid_deriv(activations[-1])
    
    for i in reversed(range(len(weights))):
        grads_w[i] = np.dot(activations[i].T, error)
        grads_b[i] = np.sum(error, axis=0, keepdims=True)
        if i != 0:
            error = np.dot(error, weights[i].T) * sigmoid_deriv(activations[i])
    return grads_w, grads_b

def predict(X, weights, biases):
    output = forward(X, weights, biases)[-1]
    return (output > 0.5).astype(int)

# ================================
# Perceptron simples para baseline
# ================================
class Perceptron:
    def __init__(self, input_dim, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros((input_dim, 1))
        self.bias = 0

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        for _ in range(self.epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.activation(linear_output)
            errors = y - y_pred
            self.weights += self.lr * np.dot(X.T, errors)
            self.bias += self.lr * np.sum(errors)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# ================================
# Gerar dataset fictício (make_moons)
# ================================
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

# Normalizar dados
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Parâmetros
input_size = X.shape[1]
hidden_layers = [12] * 10  # 10 camadas ocultas de 12 neurônios
output_size = 1
learning_rate = 0.1
epochs = 500
k_folds = 10

# Armazenar resultados para comparação
mlp_accuracies = []
mlp_losses = []

perceptron_accuracies = []

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold = 1
plt.figure(figsize=(10,6))

for train_idx, test_idx in kf.split(X):
    print(f"\nTreinando Fold {fold}/{k_folds}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inicializar pesos MLP
    layer_sizes = [input_size] + hidden_layers + [output_size]
    weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1 for i in range(len(layer_sizes)-1)]
    biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    losses = []

    # Treinamento MLP
    for epoch in range(epochs):
        activations = forward(X_train, weights, biases)
        loss = np.mean((y_train - activations[-1]) ** 2)
        losses.append(loss)
        
        grads_w, grads_b = backpropagation(y_train, activations, weights)
        for i in range(len(weights)):
            weights[i] += learning_rate * grads_w[i]
            biases[i] += learning_rate * grads_b[i]

        if epoch % 100 == 0:
            print(f"Época {epoch}, Loss: {loss:.4f}")

    # Avaliação MLP
    y_pred_mlp = predict(X_test, weights, biases)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    mlp_accuracies.append(acc_mlp)
    mlp_losses.append(losses)

    print(f"Acurácia MLP Fold {fold}: {acc_mlp * 100:.2f}%")

    # Treinamento e avaliação Perceptron simples (baseline)
    p = Perceptron(input_dim=input_size, lr=learning_rate, epochs=epochs)
    p.fit(X_train, y_train)
    y_pred_p = p.predict(X_test)
    acc_p = accuracy_score(y_test, y_pred_p)
    perceptron_accuracies.append(acc_p)
    print(f"Acurácia Perceptron Fold {fold}: {acc_p * 100:.2f}%")

    # Plot perda MLP do fold
    plt.plot(losses, label=f'Fold {fold}')
    fold += 1

# Resultados finais
print("\n=== Resultados Finais ===")
print(f"Acurácia média MLP: {np.mean(mlp_accuracies)*100:.2f}% ± {np.std(mlp_accuracies)*100:.2f}%")
print(f"Acurácia média Perceptron: {np.mean(perceptron_accuracies)*100:.2f}% ± {np.std(perceptron_accuracies)*100:.2f}%")

# Plot curvas de perda
plt.title("Curva de Erro Quadrático Médio por Fold (MLP)")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()

# Matriz de confusão do último fold MLP
cm = confusion_matrix(y_test, y_pred_mlp)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Classe 0", "Classe 1"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - Último Fold (MLP)")
plt.show()

# Relatório simples (print resumo)
print("\nRelatório Resumido:")
print(f"- Dataset: make_moons (1000 amostras, ruído=0.2)")
print(f"- Arquitetura MLP: {len(hidden_layers)} camadas ocultas de {hidden_layers[0]} neurônios cada")
print(f"- Função de ativação: sigmoid")
print(f"- Otimização: backpropagation manual, LR={learning_rate}, epochs={epochs}")
print(f"- Avaliação: {k_folds}-fold cross-validation")
print(f"- Métricas: Acurácia e Erro Quadrático Médio (MSE)")
print(f"- Resultados: MLP supera Perceptron simples em acurácia média (~{np.mean(mlp_accuracies)*100:.2f}% vs {np.mean(perceptron_accuracies)*100:.2f}%)")

