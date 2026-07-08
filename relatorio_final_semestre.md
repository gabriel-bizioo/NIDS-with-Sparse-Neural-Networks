# Relatório de Atividades - Fim de Semestre

## 1. Introdução e Objetivos do Semestre
Neste semestre, as atividades concentraram-se no estudo da base teórica sobre redes neurais aplicadas a NIDS (Network Intrusion Detection Systems) e no funcionamento interno do algoritmo SLIDE (Locality Sensitive Hashing), em paralelo com a configuração do ambiente de testes. Para a execução prática, o ambiente foi configurado na máquina `t101`, onde foi possível rodar os primeiros testes com o SLIDE. 

Contudo, essa etapa inicial apresentou algumas travas de infraestrutura: a tentativa de compilação no servidor `aura` falhou pela ausência de dependências básicas como o CMake, e na própria `t101` o desempenho foi limitado pela falta de HugePages habilitadas, o que reduziu a performance de execução em cerca de 30%. Outro contratempo foi a indisponibilidade repentina do repositório contendo a versão otimizada do SLIDE para CPU (SLIDE_opt_ia), o que direcionou os testes seguintes para a versão original do algoritmo.

---

## 2. Revisão Bibliográfica e Estudo de Datasets

### 2.1 Revisão Bibliográfica e Estudos Teóricos
A base teórica do projeto foi construída a partir da leitura e análise de trabalhos específicos que discutem a aplicação de aprendizado de máquina em segurança de redes:
Arquiteturas e Comportamento Temporal: Avaliou-se o uso de redes baseadas em Auto-encoders para detecção de anomalias com base no artigo ["An effective autoencoder-based network intrusion detection system..."](https://link.springer.com/article/10.1007/s10489-024-05872-6). Além disso, o trabalho ["Sliding Time Window for Network Intrusion Detection..."](https://arxiv.org/html/2410.18658v2) fundamentou o entendimento sobre como janelas temporais deslizantes podem ser usadas para capturar a natureza sequencial das conexões de rede, sugerindo que históricos de eventos (como filas de cabeçalhos consecutivos) podem ser estruturados de forma esparsa para alimentar o classificador.
Definição de Redes e Hiperparâmetros: Para a modelagem da rede neural, utilizou-se uma survey abrangente sobre arquiteturas de redes aplicadas a NIDS (disponível em [TechRxiv](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.175753732.26052568/v1)) e o artigo voltado para otimização de hiperparâmetros ["Hyper-parameter Optimization..."](https://arxiv.org/html/2410.22854v2). Esses estudos orientaram as decisões sobre tamanhos de camadas ocultas e taxas de aprendizado.
Pré-processamento e Modelos Hierárquicos: Artigos focados em pré-processamento de dados (como o guia de [Data Preprocessing for Neural Networks](https://www.numberanalytics.com/blog/ultimate-guide-data-preprocessing-neural-networks)) e em estruturas de classificação hierárquica auxiliaram no planejamento da formatação das entradas para a rede neural.

### 2.2 Viabilidade do SLIDE e a Importância da Esparsidade
Um ponto central discutido na literatura e avaliado ao longo do semestre é a viabilidade do uso do algoritmo SLIDE no contexto de NIDS em produção. 

O SLIDE utiliza hashing sensível à localidade (LSH) para calcular as ativações de forma esparsa, atualizando apenas uma fração dos neurônios a cada iteração. Esse mecanismo, no entanto, introduz um custo computacional fixo (o overhead de computar as funções de hash do LSH). Para que o SLIDE seja computacionalmente vantajoso frente a uma execução tradicional em CPU, a entrada fornecida à rede neural precisa ser altamente esparsa. 

Se os dados de tráfego de rede não forem adequadamente esparcificados no pré-processamento, o custo para calcular os hashes supera o ganho obtido com o treino esparso. Portanto, a modelagem das features dos pacotes deve ser feita de forma a gerar vetores esparsos de alta dimensão, viabilizando uma extração rápida em tempo real em cenários de produção.

### 2.3 Avaliação e Seleção de Datasets
Com base no levantamento feito na survey de datasets de NIDS ([arXiv:2502.06688](https://arxiv.org/abs/2502.06688)), foram estudadas diferentes bases de dados históricas e suas limitações. 

Para os testes iniciais na máquina de desenvolvimento local, foram selecionados os datasets CIC-IDS2017 e UNSW-NB15. Ambos contam com uma variedade moderna de tráfego benigno e vetores de ataque reais. O principal desafio técnico estabelecido a partir dessa escolha foi justamente aplicar técnicas de esparcificação nos atributos dessas duas bases para adequá-las às exigências de entrada do SLIDE, sem perder a capacidade do modelo de distinguir tráfego legítimo de anomalias.

## 3. Processamento de Dados e Testes Preliminares

### 3.1 Estratégia de Esparcificação de Atributos
Para que os dados de rede pudessem ser consumidos com eficiência pelo SLIDE, foi necessário planejar uma estratégia de esparcificação para os datasets CIC-IDS2017 e UNSW-NB15. 

A abordagem inicial consistiu em transformar atributos categóricos e campos estruturados de pacotes em representações esparsas (como vetores *one-hot* de grande dimensão contendo majoritariamente zeros). Nos primeiros testes com o CIC-IDS2017, o pipeline de dados foi simplificado para utilizar predominantemente os campos de **Protocolo** e **Dst Port Number (Porta de Destino)**. Essa escolha visava validar a integridade do processo de esparcificação antes de escalar para atributos mais complexos.

### 3.2 Testes de Arquitetura (Camadas Ocultas)
Paralelamente ao processamento de dados, iniciou-se a varredura para definir as dimensões apropriadas da rede neural. Foram testados diferentes tamanhos para as camadas ocultas (*hidden layers*), especificamente arquiteturas contendo:
32, 128, 256 e 1024 neurônios

Os testes práticos foram executados localmente para aferir o comportamento do tempo de treino de cada variação conforme o dataset era alimentado.

### 3.3 Resultados Obtidos e Discussão Técnica
Os experimentos preliminares trouxeram dados valiosos sobre o comportamento do modelo com entradas esparsas, evidenciando a complexidade do problema:
 - Desempenho no CIC-IDS2017 (64%):
    Na primeira versão esparsificada utilizando apenas protocolo e porta de destino, o modelo obteve uma acurácia de apenas 64%. Esse valor indica um provável subajuste (underfitting) devido à falta de atributos descritivos do tráfego. Além disso, traz o risco de overfitting a atributos específicos, onde a rede pode simplesmente memorizar associações estáticas de portas a ataques, perdendo a capacidade de generalizar padrões quando as portas mudam.
 - Desempenho no UNSW-NB15 (70%):
    Ao expandir os testes e tentar incorporar outras features para a esparcificação no dataset UNSW-NB15, a acurácia máxima alcançada foi de 70%. 
 - Limitação no Progresso:
    O fato de a acurácia ter ficado estagnada nessa faixa (entre 64% e 70% em ambos os datasets) demonstra que os pipelines de processamento testados até o momento ainda não são ideais. A esparcificação de atributos contínuos (como tamanho de pacotes e taxas de transmissão) sem degradar a capacidade preditiva da rede provou-se o principal gargalo técnico, evidenciando que simplificar demais os dados compromete a segurança do classificador, enquanto esparcificá-los de forma incorreta gera perda de informação útil.

## 4. Dificuldades Encontradas e Próximos Passos

### 4.1 Dificuldades Encontradas e Análise de Viabilidade
As principais dificuldades encontradas neste período centraram-se no gargalo de acurácia e na discussão sobre a viabilidade prática do SLIDE em comparação com abordagens tradicionais de treinamento:
O Gargalo de Acurácia e Perda de Informação:
   - Os primeiros testes mostraram que limitar a entrada do modelo a dados puramente esparsificados (como *one-hot encoding* de portas e protocolos) limita drasticamente o poder preditivo do classificador, estagnando entre 64% e 70% de acurácia. 
   - Para detectar ataques complexos, atributos contínuos (como o volume de bytes enviados, tempo entre pacotes e estatísticas de fluxo) são fundamentais. No entanto, esparsificar esse tipo de dado contínuo para alimentar o SLIDE sem perder a correlação estatística entre eles provou-se extremamente difícil. Em contrapartida, algoritmos de treinamento densos tradicionais (como Adam ou SGD padrão) processam essas variáveis contínuas de forma nativa e eficiente, atingindo acurácias tipicamente muito superiores.
SLIDE vs. Treinamento Tradicional:
   - O algoritmo SLIDE baseia-se em Hashing Sensível à Localidade (LSH) para evitar o cálculo de todas as conexões da rede, o que introduz um overhead computacional fixo para calcular os hashes. Esse mecanismo é utilizado tanto no treinamento quanto na inferência (tempo de defesa).
   - Embora o overhead do LSH seja um ponto crítico, existe a perspectiva de que a utilização de dados esparsificados simplifique significativamente a extração de features do tráfego de rede em tempo de defesa. Atributos estruturais propícios para representação esparsa (como flags de cabeçalho ou portas) são computacionalmente muito mais baratos de extrair em tempo real. Isso contrasta com o cálculo clássico de estatísticas contínuas e densas de fluxo na janela de tempo (como médias e variâncias acumuladas de bytes ou pacotes), que sobrecarregam o pipeline de inferência. A hipótese do projeto é que a aplicação de janelas temporais baseadas puramente no acúmulo de eventos esparsificados em uma consiga capturar o aspecto temporal sem gerar o overhead computacional dessas estatísticas contínuas. 
   - Desse modo, a simplificação da etapa de extração na pipeline de produção pode compensar o overhead de processamento do LSH, tornando a solução baseada em SLIDE altamente eficiente e competitiva para execução puramente em CPU, equiparando-se ou superando o desempenho de algoritmos densos tradicionais que demandam processamento de dados mais complexos na entrada.
Limitações de Infraestrutura:
   - O servidor `aura` não pôde ser aproveitado devido à ausência do CMake para compilação local. 
   - A indisponibilidade repentina do repositório contendo a versão otimizada do SLIDE para CPU (SLIDE_opt_ia) limitou a avaliação do algoritmo apenas à sua versão original, menos otimizada.

### 4.2 Próximos Passos
Para dar sequência ao projeto e elevar o patamar de acurácia do classificador, as seguintes ações estão planejadas:
Refinamento do Pipeline de Dados e Esparcificação:
   - Estudar e testar novas formas de processamento que permitam incluir features de rede contínuas (como tamanho e tempo de intervalo dos pacotes) no formato de vetores esparsificados de alta dimensão, buscando não degradar a acurácia.
   - Investigar e planejar o fluxo de extração de atributos para rodar de forma síncrona com o classificador, pensando na viabilidade técnica de rodar o modelo em ambiente real ("produção").
Exploração de Hiperparâmetros e Arquiteturas:
   - Realizar experimentos mais amplos de arquitetura de rede, focando em camadas ocultas mais largas (como 1024 neurônios) onde o SLIDE teoricamente apresenta maior eficiência computacional, buscando superar a barreira dos 70% de acurácia obtida até o momento.
