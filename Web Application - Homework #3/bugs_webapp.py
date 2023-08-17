import streamlit as st
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from helper_bugs import *

sns.set()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('/Users/lmschwenke/Downloads/bugs/uploaded', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 1   

def main():
    # Capture the file the user uploads
    uploaded_file = st.file_uploader("Upload Image")

    # Set page variable equal to the model the user selects
    page=st.sidebar.selectbox(""" Please select a model:""", 
         ["Custom CNN Model", 
          "ResNet-50 Model"])

    if page == "Custom CNN Model":
        st.title("Bug Classifier using Custom CNN Model")

        if uploaded_file is not None:
        #logging.info("File uploaded: %s", uploaded_file.name)
            if save_uploaded_file(uploaded_file):
                # Display the file
                display_image = Image.open(uploaded_file)
                display_image = display_image.resize((500, 300))
                st.image(display_image)

                prediction = predictor(os.path.join('uploaded', uploaded_file.name), predictor_model_cnn)
                logging.info("Prediction: %s", prediction)

                # Drawing graphs
                st.markdown('**Predictions**')
                fig, ax = plt.subplots()
                ax = sns.barplot(y='name', x='values', data=prediction, order=prediction.sort_values('values', ascending=False).name)
                ax.set(xlabel='Confidence %', ylabel='Breed')
                st.pyplot(fig)

    else:
     ### INFO
        st.title("Bug Classifier using ResNet-50 Model")

        if uploaded_file is not None:
        #logging.info("File uploaded: %s", uploaded_file.name)
            if save_uploaded_file(uploaded_file):
                # Display the file
                display_image = Image.open(uploaded_file)
                display_image = display_image.resize((500, 300))
                st.image(display_image)

                prediction = predictor(os.path.join('uploaded', uploaded_file.name), predictor_model_resnet)
                logging.info("Prediction: %s", prediction)

                # Drawing graphs
                st.markdown('**Predictions**')
                fig, ax = plt.subplots()
                ax = sns.barplot(y='name', x='values', data=prediction, order=prediction.sort_values('values', ascending=False).name)
                ax.set(xlabel='Confidence %', ylabel='Breed')
                st.pyplot(fig)


if __name__ == '__main__':
    main()
