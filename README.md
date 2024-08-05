# Vision-to-Speech Project
This project utilizes a pre-trained model to derive actions from a video input and output speech descriptions of these actions. The pipeline involves video processing, action description generation, semantic similarity filtering to ensure unique descriptions, and speech synthesis.


### Overview
The Vision-to-Speech project aims to convert video inputs into spoken descriptions. It processes each frame of the video, generates descriptive captions using a pre-trained BLIP model, filters out redundant descriptions, translates the text if necessary, and converts the final description to speech.

### Features
Frame to Text: Converts video frames to descriptive text using a pre-trained BLIP model.
Unique Meaning Extraction: Filters out similar text descriptions to retain only unique meanings using semantic similarity.
Translation: Translates text descriptions into a specified language using Google Translate.
Text to Speech: Converts the text descriptions into speech using Google Text-to-Speech (gTTS).

### Requirements
Python 3.7 or higher

Libraries:

- os
- cv2 (OpenCV)
- PIL (Pillow)
- transformers
- sentence_transformers
- googletrans
- gtts

### Setup and Installation

##### Clone the repository:
##### Install the required libraries:
- pip install opencv-python
- pip install Pillow
- pip install transformers
- pip install sentence-transformers
- pip install googletrans==4.0.0-rc1
- pip install gtts

### Usage
Prepare your video file and place it in the project directory.    
Update the video_path variable in the script with the path to your video file.

Run the script:
***** replace with streamlit app name here ***** (python vision_to_speech.py)

### Testing the Model
To test the accuracy of the model, you can manually compare the generated descriptions with the actual video content. 


### Hosting the Application

