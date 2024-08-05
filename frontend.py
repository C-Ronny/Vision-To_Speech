import os
import cv2
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
from googletrans import Translator


# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the semantic similarity model
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize the translator
translator = Translator()

def frame_to_text(frame):
    """Convert a single frame to text using the model."""
    image = Image.fromarray(frame)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

def get_unique_meanings(texts, threshold=0.8):
    """Filter out texts that have similar meanings based on semantic similarity."""
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

def video_to_text(video_path, repeat_threshold=5):
    """Process video frames and generate meaningful text descriptions based on repeated actions."""
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

def translate_text(text, target_lang='ha'):
    translation = translator.translate(text, dest=target_lang)
    return translation.text

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("afplay output.mp3")

# Streamlit app layout
st.set_page_config(page_title="Video to Text Description", layout="wide")

st.title("Video to Text Description and Translation")
st.write("Upload a video file to generate a text description, translate it, and convert it to speech.")

# Sidebar for file upload and options
st.sidebar.header("Upload and Options")
uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
target_language = st.sidebar.selectbox("Select target language for translation", ['ha', 'en', 'es', 'fr'])

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

                if st.sidebar.button("Play Audio"):
                    text_to_speech(translated_description, lang=target_language)
                    audio_file = open("output.mp3", "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
else:
    st.info("Please upload a video file to start.")

# Main content area
st.header("Instructions")
st.write("""
1. Upload a video file using the sidebar.
2. Select the target language for translation.
3. Click on 'Generate Description' to process the video and generate text.
4. The translated description will be displayed below the original description.
5. Click on 'Play Audio' to listen to the translated description.
""")