import streamlit as st
import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import tensorflow as tf
from googletrans import Translator
from gtts import gTTS

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the processor and model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the semantic similarity model from .h5 file
semantic_model = tf.keras.models.load_model('semantic_model.h5')

# Function to convert frame to text
def frame_to_text(frame):
    image = Image.fromarray(frame)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

# Function to filter out similar text descriptions
def get_unique_meanings(texts, threshold=0.75):
    unique_texts = []
    for text in texts:
        text_embedding = semantic_model.predict([text])
        is_unique = True
        for unique_text in unique_texts:
            unique_text_embedding = semantic_model.predict([unique_text])
            similarity = np.dot(text_embedding, unique_text_embedding.T).item()
            if similarity > threshold:
                is_unique = False
                break
        if is_unique:
            unique_texts.append(text)
    return unique_texts

# Function to process video and generate meaningful text descriptions
def video_to_text(video_path, repeat_threshold=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return ""
    current_action = None
    action_count = 0
    frame_texts = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        text = frame_to_text(frame)
        if text == current_action:
            action_count += 1
        else:
            if action_count >= repeat_threshold and current_action is not None:
                frame_texts.append(f"There was '{current_action}'.")
            current_action = text
            action_count = 1
    if action_count >= repeat_threshold and current_action is not None:
        frame_texts.append(f"The action '{current_action}'.")
    cap.release()
    unique_frame_texts = get_unique_meanings(frame_texts)
    video_description = " ".join(unique_frame_texts)
    return video_description

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("afplay output.mp3")

# Function to translate text
def translate_text(text, target_lang='ha'):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text

# Streamlit UI
st.title("Video to Speech Application")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.getbuffer())
    description = video_to_text("temp_video.mp4")
    st.write("Original Description:", description)
    translated_description = translate_text(description, target_lang='ha')
    st.write("Translated Description:", translated_description)
    text_to_speech(translated_description, lang='ha')
    st.audio("output.mp3")
