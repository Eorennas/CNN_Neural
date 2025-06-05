import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests


model = tf.keras.applications.MobileNetV2(weights='imagenet')


def prepare_image(image):
    print(f'Imagem original: {image}')  
    image = image.resize((224, 224)) 
    print(f'Imagem redimensionada: {image.size}')  
    image = np.array(image)  
    print(f'Imagem como array: {image.shape}')  
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image) 
    print(f'Imagem preprocessada: {image.shape}')
    image = np.expand_dims(image, axis=0) 
    return image


def predict_image(image):
    prepared_image = prepare_image(image)
    print(f'Imagem preparada para predição: {prepared_image.shape}')  
    predictions = model.predict(prepared_image)
    print(f'Previsões: {predictions}')
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0] 
    return decoded_predictions


def display_statistics(predictions):
    st.header("Estatísticas de Acurácia e Classes Identificadas")

    st.subheader("Top 5 Classes Identificadas:")
    for i, pred in enumerate(predictions):
        st.write(f"{i + 1}. {pred[1]} com probabilidade de {pred[2] * 100:.2f}%")

  
    st.subheader("Acurácia do Modelo:")
    st.write(
        "A acurácia geral do modelo em tarefas de classificação é aproximadamente 71.8% (valores podem variar dependendo da classe).")

   
    labels = [pred[1] for pred in predictions]
    probabilities = [pred[2] for pred in predictions]

    fig, ax = plt.subplots()
    ax.barh(labels, probabilities, color='skyblue')
    ax.set_xlabel('Probabilidade')
    ax.set_title('Top 5 Classes Identificadas')
    plt.tight_layout()
    st.pyplot(fig)


def load_imagenet_classes():
    imagenet_classes = {}
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(url)
    imagenet_classes = response.json()

   
    class_data = {
        "Classe ID": [key for key in imagenet_classes],
        "Classe": [value[1] for value in imagenet_classes.values()],
        "Descrição": [value[0] for value in imagenet_classes.values()],
    }
    return pd.DataFrame(class_data)


imagenet_df = load_imagenet_classes()


st.title('Classificador de Objetos em Imagens')
st.write(' Elaborado por Rennan Alves, Rodrigo Medeiros e Guilherme Silva. ')

st.write("""
    Este classificador foi desenvolvidor para uma experiencia de imersão com uma rede neural para classificar objetos.
    O modelo de classificação tenta identificar as classes mais prováveis e exibir as probabilidades de cada classe.
    Ao carregar uma imagem, o modelo tentará identificar o objeto presente nela, mostrando o nome da classe e a probabilidade de acerto.
    Além disso, você poderá ver as **estatísticas de acurácia** do modelo e as **top 5 classes** mais próximas identificadas.
""")


tabs = st.radio("Escolha uma seção:", ("Identificação de Objeto", "Estatísticas de Acurácia", "Tabela de Classes"))


if tabs == "Identificação de Objeto":
    st.write("Carregue uma imagem para que o modelo identifique o objeto presente nela.")

   
    uploaded_image = st.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
       
        image = Image.open(uploaded_image)
        st.image(image, caption="Imagem carregada", use_container_width=True)

        result = predict_image(image)

      
        st.write(f"Identificado: {result[0][1]} com uma probabilidade de {result[0][2] * 100:.2f}%")
        display_statistics(result)


elif tabs == "Estatísticas de Acurácia":
    st.header("Acurácia do Modelo")
    st.write("A acurácia do modelo em tarefas de classificação é aproximadamente 71.8%.")
    st.write(
        "Isso significa que o modelo é capaz de classificar corretamente cerca de 71.8% das imagens em um grande conjunto de categorias.")
    st.write("Esse número é uma média e pode variar dependendo da categoria ou do tipo de imagem.")


# Seção de Classes Mais Próximas
elif tabs == "Classes Mais Próximas":
    st.header("Top 5 Classes Mais Próximas Identificadas")
    st.write(
        "O modelo pode identificar várias classes. Aqui estão as top 5 classes mais prováveis com base nas últimas imagens classificadas:")

    # Essa seção mostrará os top 5 resultados da última imagem carregada
    if uploaded_image is not None:
        predictions = predict_image(image)
        display_statistics(predictions)

# Seção de Tabela de Classes
elif tabs == "Tabela de Classes":
    st.header("Tabela de Classes Existentes")
    st.write(""" Aqui estão todas as classes identificadas pelo modelo. A tabela mostra o ID, o nome da classe e uma descrição simples. """)

    # Exibe a tabela de classes
    st.dataframe(imagenet_df)
