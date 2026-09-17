 Rede Neural — Classificador de Pessoas

Mini projeto de rede neural criado com TensorFlow.js que classifica pessoas 
em categorias premium, medium ou basic com base em idade, 
cor favorita e localização.

 O que o projeto faz

A rede neural é treinada com dados de 3 pessoas e aprende a identificar 
padrões. Após o treinamento, ela recebe uma nova pessoa e retorna a 
probabilidade de cada categoria.

Exemplo de saída:

basic (64.71%)
premium (24.86%)
medium (10.43%)


 Arquivos

- `index.js` — código principal da rede neural
- `*.json` — dados utilizados no projeto

 Como rodar

1. Instale as dependências:
```bash
npm install @tensorflow/tfjs
```

2. Execute o arquivo:
```bash
node index.js
```

 O que aprendi

- Conceito de redes neurais e como elas aprendem com exemplos
- Como treinar um modelo do zero com poucos dados
- One-hot encoding — transformar dados como cor e localização em números
- Normalização de dados numéricos como idade
- Como usar TensorFlow.js para criar e treinar modelos em JavaScript

 Tecnologias

- JavaScript
- TensorFlow.js