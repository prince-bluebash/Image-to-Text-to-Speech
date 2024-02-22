import streamlit as st
from PIL import Image
from transformers import pipeline
from langchain_openai import ChatOpenAI
import pdb
import wave
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

openai_key = st.sidebar.text_input("Enter your OpenAI Key", type="password")
# Use a pipeline as a high-level helper

pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
pipe2 = pipeline("text-to-speech", model="facebook/mms-tts-eng")
output_parser = StrOutputParser()

def get_scenario_from_img_text(text):
    llm = ChatOpenAI(openai_api_key=openai_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You need to write few lines on the scenario provided by user."),
        ("user", "{text}")
    ])
    chain = prompt | llm 
    response = chain.invoke({"text": text})
    return  response.content

def main():
    st.title("Image-to-Text-to-Speech App")

    # Upload image file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Generate text from the image
        result = pipe(img)
        text = result[0]['generated_text']
        print(text)
        text = get_scenario_from_img_text(text)
        print(text)
        speech = speech = pipe2(text)
        # Display the generated text
        st.subheader("Generated Text:")
        st.write(text)
        wav_data = np.array(speech['audio'] * 32767, dtype=np.int16)
        with wave.open('output.wav', 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(wav_data.dtype.itemsize)
            wav_file.setframerate(speech['sampling_rate'])
            wav_file.writeframes(wav_data.tobytes())

        # Play the WAV file using the st.audio() function
        st.audio('output.wav')

if __name__ == "__main__":
    main()