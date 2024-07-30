import streamlit as st
from micrograd.schemas.schemas import ModelSchema
from micrograd.nets.NeuralNet import NeuralNet
from micrograd.engine.value import Value
from PIL import Image
import numpy as np
import json

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    weights_file = "model_weights_0-79.json"
    model = load_model(weights_file)

    # Prepare the streamlit interface
    st.title("The World's Worst Hotdog Classifier")
    st.write("Hey! My name is Arnie. I'm a MSc CS student @ ETH Zurich. 👋 Here's my [Twitter](https://x.com/arnie_hacker) & [LinkedIn](https://www.linkedin.com/in/anirudhhramesh/) ")
    st.subheader("What is this?")
    st.write("Inspired by Jian-Yang from the show Silicon Valley, I decided to build a hotdog classifier.")
    st.write("My classifier has a training loss of 0.8 and a training accuracy of 52%. Making it quite possibly the world's worst hotdog classifier :)")

    st.image("jian_yang_hotdog.png", caption="Jian-Yang's hotdog classifier, from the show Silicon Valley", width=600)
    
    st.subheader("So why is this cool anyway?")
    st.write("1. I built the classifier *entirely* from scratch. No ML libraries, no existing models.")
    st.write("2. I did not follow any tutorials. After finishing @DeepLearning.ai's ML Specialization, I built this through first principles reasoning, deriving the backpropagation algorithm from scratch, thinking through e.g. chain rule, neural net design, ...")
    st.write("3. I learnt ML & built this project in evenings & weekends, while juggling a full-time software internship at Amazon Web Services (and doing sports 4x per week)! . ")
    
    st.write("Finishing the ML specialization & building this fun project (and more random coding & sports) basically led to me working ~10-12hrs per day for the past 2 months :) ")
    st.write("But I'm not at all burnt out! I'm so happy to be able to finally learn and get through all these concepts through pure curiousity. The past semesters I had spent building a start-up (even interviewing for YCombinator) but:")
    st.write("- In my YC interview question, they asked the simple question 'why do you want to build [this](https://app.swirl.so)?'. And I couldn't answer it. I realized I hadn't figured out what I purposeful thing I wanted to work on for the next 10+ years")
    st.write("- After getting on Twitter around March 2024, I had a growing bucket list of cool projects I wanted to build. Machine Learning, Robotics, LLMs, .... It felt so amazing to finally have time to work on these projects (with no deadlines or limitations!)")
    
    st.write("If this impresses you or you find it funny - go follow me on [Twitter](https://x.com/arnie_hacker) (where I document my daily grind) and [LinkedIn](https://www.linkedin.com/in/anirudhhramesh/)!")

    st.write("Here is the [code](https://github.com/AnirudhhRamesh/micrograd) if you want to see the behind-the-scenes or run it locally!")

    st.subheader("Test out the model yourself! 👇")
    
    # Display images 1-4 in the same row
    col1, col2 = st.columns(2)
    with col1:
        st.image("hotdog1.jpg", width=200, caption="hotdog1.jpg")
        st.image("hotdog2.jpg", width=200, caption="hotdog2.jpg")
    with col2:
        st.image("not_hotdog1.jpg", width=200, caption="not_hotdog1.jpg")
        st.image("not_hotdog2.jpg", width=200, caption="not_hotdog2.jpg")

    # Select between 4 images
    selected_image = st.selectbox("Select an image", options=["hotdog1.jpg", "hotdog2.jpg", "not_hotdog1.jpg", "not_hotdog2.jpg"])
    
    # Display the selected image
    st.image(selected_image, width=200)

    # A button which predicts the image
    if st.button("Predict"):
        st.write(f"Predicting {selected_image}...")
        st.write(f"{predict(selected_image, model)}")
    
def load_model(file:str):
    # Parse weights from weights.json
    logger.info(f"Loading model weights from {file}...")
    with open(file, 'r') as f:
        model_json = f.read()
    
    validated_model = ModelSchema.model_validate_json(model_json)
    
    model = NeuralNet.from_json(validated_model)
    logger.info(f"Model loaded from {file}")
    
    return model

def predict(image:str, model):
    prediction = model(convert(image))

    predicted = "Hotdog" if prediction.value > 0.5 else "Not Hotdog"
    return f"Prediction: {predicted}, Confidence: {prediction.value:.2f}"

def convert(image_path, image_size=225):
    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    image = image.convert('L')  # Convert to grayscale
    img_array = np.array(image).flatten() / 255.0  # Normalize to [0, 1]
    return img_array

def render():
    st.write("Hello World")

if __name__ == "__main__":
    main()