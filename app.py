import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from googletrans import Translator  # Biblioteca para tradução automática

# Carregar um modelo pré-treinado para identificação de objetos
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Função para preparar a imagem antes de passar para o modelo
def prepare_image(image):
    image = image.resize((224, 224))  # Redimensiona para 224x224
    image = np.array(image)  # Converte a imagem para um array numpy
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Pré-processamento
    image = np.expand_dims(image, axis=0)  # Adiciona uma dimensão extra para o batch
    return image

# Função para classificar a imagem
def predict_image(image):
    # Prepara a imagem
    prepared_image = prepare_image(image)
    
    # Faz a previsão
    predictions = model.predict(prepared_image)
    
    # Decodifica a previsão
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]  # Top 5 previsões
    
    return decoded_predictions

# Função para exibir estatísticas
def display_statistics(predictions):
    st.header("Estatísticas de Acurácia e Classes Identificadas")
    
    st.subheader("Top 5 Classes Identificadas:")
    for i, pred in enumerate(predictions):
        st.write(f"{i+1}. {pred[1]} com probabilidade de {pred[2]*100:.2f}%")

    # Exemplo de acurácia fictícia (pode ser substituído por dados reais de acurácia de teste do modelo)
    st.subheader("Acurácia do Modelo:")
    st.write("A acurácia geral do modelo em tarefas de classificação é aproximadamente 71.8% (valores podem variar dependendo da classe).")

    # Gerando gráfico das probabilidades
    labels = [pred[1] for pred in predictions]
    probabilities = [pred[2] for pred in predictions]
    
    fig, ax = plt.subplots()
    ax.barh(labels, probabilities, color='skyblue')
    ax.set_xlabel('Probabilidade')
    ax.set_title('Top 5 Classes Identificadas')
    plt.tight_layout()
    st.pyplot(fig)

# Função para carregar todas as classes do ImageNet
def load_imagenet_classes():
    # A lista de classes do ImageNet (ID, Nome de Classe)
    imagenet_classes = {}
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    import requests
    response = requests.get(url)
    imagenet_classes = response.json()
    
    # Convertendo em um DataFrame para facilitar visualização
    class_data = {
        "Classe ID": [key for key in imagenet_classes],
        "Classe": [value[1] for value in imagenet_classes.values()],
        "Descrição": [value[0] for value in imagenet_classes.values()],
    }
    return pd.DataFrame(class_data)

# Função para traduzir classes e descrições para o português
def translate_to_portuguese(df):
    translator = Translator()
    df['Classe'] = df['Classe'].apply(lambda x: translator.translate(x, src='en', dest='pt').text)
    df['Descrição'] = df['Descrição'].apply(lambda x: translator.translate(x, src='en', dest='pt').text)
    return df

# Carregando todas as classes do ImageNet
imagenet_df = load_imagenet_classes()

# Traduzindo as classes e descrições para o português
imagenet_df = translate_to_portuguese(imagenet_df)

# Título da aplicação
st.title('Classificador de Objetos em Imagens')

# Descrição sobre o que é um classificador de objetos
st.write("""
    Este aplicativo permite carregar uma imagem para identificar objetos presentes nela. O modelo de classificação tenta 
    identificar as classes mais prováveis e exibir as probabilidades de cada classe.
    
    Ao carregar uma imagem, o modelo tentará identificar o objeto presente nela, mostrando o nome da classe e a probabilidade de acerto.
    Além disso, você poderá ver as **estatísticas de acurácia** do modelo e as **top 5 classes** mais próximas identificadas.
""")

# Criação de abas para o usuário
tabs = st.radio("Escolha uma seção:", ("Identificação de Objeto", "Estatísticas de Acurácia", "Classes Mais Próximas", "Tabela de Classes"))

# Seção de Identificação de Objeto
if tabs == "Identificação de Objeto":
    st.write("Carregue uma imagem para que o modelo identifique o objeto presente nela.")
    
    # Upload da imagem
    uploaded_image = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Exibe a imagem carregada com o parâmetro atualizado
        image = Image.open(uploaded_image)
        st.image(image, caption="Imagem carregada", use_container_width=True)
        
        # Realiza a predição
        result = predict_image(image)
        
        # Exibe o resultado da predição
        st.write(f"Identificado: {result[0][1]} com uma probabilidade de {result[0][2]*100:.2f}%")
        display_statistics(result)  # Exibe as estatísticas de acurácia e as top 5 classes

# Seção de Estatísticas de Acurácia
elif tabs == "Estatísticas de Acurácia":
    st.header("Acurácia do Modelo")
    st.write("A acurácia do modelo em tarefas de classificação é aproximadamente 71.8%.")
    st.write("Isso significa que o modelo é capaz de classificar corretamente cerca de 71.8% das imagens em um grande conjunto de categorias.")
    st.write("Esse número é uma média e pode variar dependendo da categoria ou do tipo de imagem.")
    
    st.subheader("Acurácia por classe:")
    st.write("Infelizmente, não temos dados específicos de acurácia por classe aqui, mas você pode consultar o desempenho do modelo em tarefas específicas de classificação.")

# Seção de Classes Mais Próximas
elif tabs == "Classes Mais Próximas":
    st.header("Top 5 Classes Mais Próximas Identificadas")
    st.write("O modelo pode identificar várias classes. Aqui estão as top 5 classes mais prováveis com base nas últimas imagens classificadas:")
    # Essa seção mostrará os top 5 resultados da última imagem carregada
    if uploaded_image is not None:
        predictions = predict_image(image)
        display_statistics(predictions)

# Seção de Tabela de Classes
elif tabs == "Tabela de Classes":
    st.header("Tabela de Classes Existentes")
    st.write("""
        Aqui estão todas as classes identificadas pelo modelo. A tabela mostra o ID, o nome da classe e uma descrição simples.
    """)
    
    # Exibe a tabela de classes
    st.dataframe(imagenet_df)
