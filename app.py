import streamlit as st
import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator
from gtts import gTTS
import numpy as np
from tempfile import NamedTemporaryFile

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the image captioning model and processor from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the semantic similarity model directly from Hugging Face
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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
        text_embedding = semantic_model.encode(text, convert_to_tensor=True)
        is_unique = True
        for unique_text in unique_texts:
            unique_text_embedding = semantic_model.encode(unique_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(text_embedding, unique_text_embedding).item()
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
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 15, dtype=int)
    
    frame_texts = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        text = frame_to_text(frame)
        frame_texts.append(text)
    
    cap.release()
    unique_frame_texts = get_unique_meanings(frame_texts)
    video_description = " ".join(unique_frame_texts)
    return video_description

# Function to convert text to speech and return the path to the audio file
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    temp_audio = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# Function to translate text
def translate_text(text, target_lang='ha'):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text

# Streamlit app layout
st.set_page_config(page_title="Video to Text Description", layout="wide")

st.title("Video to Text Description and Translation")
st.write("Upload a video file to generate a text description, translate it, and convert it to speech.")

# Sidebar for file upload and options
st.sidebar.header("Upload and Options")
uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
target_language = st.sidebar.selectbox("Select target language for translation (ha - Hausa, sn - Shona, yo -  Yoruba, en - English)", ['ha', 'sn', 'yo', 'en'])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video("temp_video.mp4")

    if st.sidebar.button("Generate Description"):
        with st.spinner("Processing..."):
            description = video_to_text("temp_video.mp4")
            if description:
                st.success("Description generated successfully!")
                st.write("Original Description:", description)

                translated_description = translate_text(description, target_lang=target_language)
                st.write(f"Translated Description ({target_language}):", translated_description)

                audio_file_path = text_to_speech(translated_description, lang=target_language)
                st.audio(audio_file_path, format="audio/mp3")
else:
    st.info("Please upload a video file to start.")

# Main content area
st.header("Instructions")
st.write("""
1. Upload a video file using the sidebar.
2. Select the target language for translation.
3. Click on 'Generate Description' to process the video and generate text.
4. The translated description will be displayed below the original description.
5. The audio will be played automatically after the translation.
""")
