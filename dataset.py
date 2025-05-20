import numpy as np
import pandas as pd

# Configuração para reproduzibilidade
np.random.seed(42)

# Gerando dados fictícios
n_samples = 500

# Idade (entre 18 e 70 anos)
idade = np.random.randint(18, 71, size=n_samples)

# Salário (entre 1000 e 15000 reais)
salario = np.random.randint(1000, 15001, size=n_samples)

# Tempo de emprego (entre 0 e 30 anos)
tempo_emprego = np.random.randint(0, 31, size=n_samples)

# Dívida atual (entre 0 e 5000 reais)
divida = np.random.randint(0, 5001, size=n_samples)

# Escolaridade (0=fundamental, 1=médio, 2=superior)
escolaridade = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2])

# Função para determinar aprovação (regras fictícias)
def calcular_aprovacao(idade, salario, tempo_emprego, divida, escolaridade):
    score = 0
    
    # Pontuação por idade (25-50 anos é melhor)
    if 25 <= idade <= 50:
        score += 1
    elif idade < 25 or idade > 60:
        score -= 0.5
    
    # Pontuação por salário
    score += min(salario / 5000, 2)  # até 2 pontos para salários altos
    
    # Pontuação por tempo de emprego
    score += min(tempo_emprego / 5, 2)  # até 2 pontos para +5 anos
    
    # Pontuação por dívida (quanto menor melhor)
    score -= min(divida / 2000, 1.5)  # penalização por dívida alta
    
    # Pontuação por escolaridade
    score += escolaridade * 0.5  # 0, 0.5 ou 1 ponto
    
    # Aprovação baseada no score
    return 1 if score > 2.5 else 0

# Gerando a coluna de aprovação
aprovado = np.array([
    calcular_aprovacao(idade[i], salario[i], tempo_emprego[i], divida[i], escolaridade[i]) 
    for i in range(n_samples)
])

# Criando o DataFrame
df = pd.DataFrame({
    'Idade': idade,
    'Salario': salario,
    'Tempo_Emprego': tempo_emprego,
    'Divida': divida,
    'Escolaridade': escolaridade,
    'Aprovado': aprovado
})

# Exibindo as primeiras linhas
print(df.head())

# Salvar para CSV (opcional)
df.to_csv('credito_dataset.csv', index=False)