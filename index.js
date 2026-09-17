import * as tf from '@tensorflow/tfjs';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    //primeira camada da rede:
    //entrda com 7 posições (idade normalizada, + 3 cores + 3 localizações)

    //80 neuronios = aqui coloquei tudo issso , pq tem pouca base de treino
    //quanto mais neuronios, mais complexidade a rede pode aprender
    //e consequentemente, mais processanento ela vai usar

    //A relu age como um filtro:
    //é como se ela deixasse somente os dados insteressantes passarem
    //se a informação chegou nesse neuronio é positivo, passa para frente!
    //se for negativa, pode jogar fora, não vai servir para nada!

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    //saida da rede, 3 neuronios, um para cada label (premium, medium, basic)

    //activation softmax: é uma função que vai pegar os 3 valores de saída e vai normalizar eles, para que a soma deles seja 1
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    //compilando o modelo, definindo a função de perda e o otimizador
    //optimizer Adam ( Adaptive Moment Estimation) 
    //é um treinador pessoal moderno para redes neurais:
    //Loss: categoricalCrossentropy
    //ele compara o que o modelo "acha" (os scores de cada categoria)
    //com a resposta correta 
    //a categoria premium será sempre [1, 0, 0] 

    //quanto mais distante da previsão do modelo da resposta correta
    //maior o erro
    //Exemplo classico: classificação de imagens, recomendação, categorização de 
    //usuario
    //qualquer coisa em que a resposta certa é apenas uma entre varias possiveis

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    //treinamento do modelo
    //verbose: desabilita o log interno ( e usa so callback)
    //epochs: quantidade de vezes que vai rodar no dataset
    //shuffle: embaralha os dados, para evitar viés

    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
          /* callbacks: {
                onEpochEnd: (epoch, log) => console.log(
                    `Epoch: ${epoch}: loss = ${log.loss}`
                )
            }*/ 
           //comentei pos gasta muitas linhas !!
        }
    )

    return model
}

async function predict ( model,pessoa){
    //tranformar o array js para o tnsor (tfjs)
    const tfinput = tf.tensor2d(pessoa)

    //faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfinput)
    const predArray = await pred.array()
return predArray[0].map((prob, index) => ({prob, index}))
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//Quanto mais dado melhor!
//assim o algoritmo  consegue entender melhor os padroes complexos dos dados

const model = await trainModel(inputXs, outputYs);

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba" }
//normalizando a idade da nova pessoa usando o mesmo padrão do treino
//Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2

const pessoaTensorNormalizado = [ // vetor de entrada para a nova pessoa
    [
        0.2, //idade normalizada
        1, // cor azul
        0, // cor vermelho
        1,  // cor verde
        1, // Localização São Paulo
        1, // Localização Rio
        0 //Localização Coritiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado)
const results = predictions
.sort((a,b) => b. prob - a.prob)
.map(p => `${labelsNomes[p.index]} (${(p.prob* 100).toFixed (2)}%)`)
.join('\n')

console.log(results)