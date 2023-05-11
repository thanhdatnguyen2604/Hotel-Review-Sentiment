import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import pickle
from processing import pipeline_processing
from sklearn.metrics import classification_report, confusion_matrix
import xlsxwriter   
from io import BytesIO

#--------------
# GUI
st.title("Machine Learning Project")
st.header("Sentiment Analysis")

# Load models 
file_name= r"D:\Code\Project-ML\Save_Sentiment_Analysis_Models\LSTM_sentiment_model_undersampling.h5"
model= tf.keras.models.load_model(file_name)
model.load_weights(r'D:\Code\Project-ML\Save_Sentiment_Analysis_Models\model_cp_rnn_under.h5')
with open(r'D:\Code\Project-ML\Hotel-Review-Sentiment\Dataset\tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

# GUI
menu = ["Introduction", "Prediction"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Introduction':    
    st.subheader("Introduction")
    st.write("""
    ###### Sentiment analysis is part of the Natural Language Processing (NLP) techniques that consists in extracting emotions related to some raw texts. This is usually used on social media posts and customer reviews in order to automatically understand if some users are positive or negative and why.
    """)  
    st.write("""###### For each textual review, we want to predict if it corresponds to a good review (the customer is happy) or to a bad one (the customer is not satisfied). The reviews overall ratings can range from 1/5 to 5/5. In order to simplify the problem we will split those into two categories:""")
    st.write("""
    ###### - Negative reviews have overall ratings <=3
    ###### - Positive reviews have overall ratings > 3
    """)
    st.image("sentiment_analysis.jpg")

elif choice == 'Prediction':
    flag = False
    data = None
    st.subheader("Upload file or Input data")
    type = st.radio(label="", options=("Upload file", "Input text"))
    if type=="Upload file":
        # Upload file
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            check= str(uploaded_file.name).split('.')[1]
            if check=='csv':
                data = pd.read_csv(uploaded_file, header=0)
            elif check=='xlsx':
                data= pd.read_excel(uploaded_file, header=0)

            if len(data)>0:
                st.subheader("Result:")

                data['text_clean']= data['text'].apply(lambda x: pipeline_processing(x))
                sequences= tok.texts_to_sequences(data['text_clean'])
                sequences_matrix= sequence.pad_sequences(sequences, maxlen= 100)               
                y_pred_new = model.predict(sequences_matrix)     
                result = [1 if prob >= 0.5 else 0 for prob in y_pred_new]  
                data['predict']= pd.DataFrame(result)

                st.code("Confusion matrix: \n" + str(confusion_matrix(data['sentiment'], data['predict'])))
                st.code("Classification Report: \n" + str(classification_report(data['sentiment'], data['predict'])))

                data['predict'] = pd.DataFrame(["Positive" if prob >= 0.5 else "Negative" for prob in y_pred_new])
                data= data[['text', 'sentiment', 'predict']]

                # Save data_new.xlsx
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                data.to_excel(writer, encoding='utf-8')
                writer.save()

                st.download_button(label="Click here to download file after predict!", 
                                    data=output.getvalue(),
                                    file_name='data_new.xlsx',
                                    mime='application/vnd.ms-excel')

    if type=="Input text":        
        text = st.text_area(label="Input your content:")
        if text != "":
            text = np.array([text])
            text = map(lambda x: pipeline_processing(x), text)
            sequences= tok.texts_to_sequences(text)
            sequences_matrix= sequence.pad_sequences(sequences, maxlen= 100)
            y_pred_new = model.predict(sequences_matrix)     
            result = ["Positive" if prob >= 0.5 else "Negative" for prob in y_pred_new] 
            st.subheader("Result:")
            st.code("Predictions [Negative, Positive]: " + str(result))

