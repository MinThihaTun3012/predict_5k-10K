#import Libraries
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import pyttsx3
from imutils.perspective import four_point_transform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input



# Set page config
st.set_page_config(
    page_title = "Money Classification",
    page_icon = "img/dollar.jpg",
    layout="centered", 
    initial_sidebar_state="expanded"
)



#<---- functions start ---->


# # Define a function to convert text to speech
# def text_to_speech(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()

# loaded the model from other path 
def loadModel():
    model = tf.keras.models.load_model("v6.h5",compile= False)
    return model

model = loadModel()


# image processing
three_notes = []

def getSquares(input):
    height = 600
    width = 800
    green = (0, 255, 0) # green color
    # red = (0, 0, 255) # red color
    # white = (255, 255, 255) # white color
    # questions = 5
    # answers = 5
    # correct_ans = [0, 2, 1, 3, 4]

    img = cv2.imread(input)
    img = cv2.resize(img, (width, height))
    img_copy = img.copy() # for display purposes

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    edge_img = cv2.Canny(blur_img, 20, 180) # adjust with 


    # find the contours in the image
    contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # get rect cnts
    rect_cnts = []
    for cnt in contours:
        # approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # if the approximated contour is a rectangle ...
        if len(approx) == 4:
            # append it to our list
            rect_cnts.append(approx)
    # sort the contours from biggest to smallest
    rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)
    
    
    for i in range(0,6,2):
        document = four_point_transform(img_copy, rect_cnts[i].reshape(4, 2))
        document = cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
        three_notes.append(document)
        st.image(document)


# <--- factions end --->



#<--- slide starting --->


st.sidebar.image("img/dollar.jpg")

# Options menu
option = st.sidebar.selectbox("Chosse what you want to do",
                            ("Limitations","Money Classification With Color Background","Money Classification With White Background"))

#<--- slide ending -->


st.markdown("<h1 style='color: #19376D;'>Classification Of Myanmar Bank Note</h1>", unsafe_allow_html= True)


if option == "Limitations":
        st.markdown("<h3 style='color: #1A5F7A;'>Limitation</h3>", unsafe_allow_html= True)
        st.write("- We can only classify five or ten thousand kyats")
        st.write("- We are able to extract exact 3 notes of currency from one input image")
        st.write("- Accuracy depends on the background of the image, more clear backgrounds make our model detect currency notes more precisely. ")
        st.write("- We will be able to calculate the total amount of money in the image")



elif option == "Money Classification With Color Background":
    st.markdown("<h3 style='color: #1A5F7A;'>Money Classification With Color Background</h3>", unsafe_allow_html= True)
    st.write("- White backgrounds make the model hard to extract the currency ")
    st.write("- Messy background will make the model a bit hard to get bank notes from the image")
    st.write("- We are able to extract exact 3 notes of currency from one input image")


    #load file
    uploaded_file = st.file_uploader("Upload Your Image",type=["png", "jpg"])

    if uploaded_file is None:
        st.success("Please upload your image")
    
    else:
        st.success("Your image has been uploaded")
        photo = Image.open(uploaded_file)
        st.image(photo, caption='Your image has been uploaded') 
        image_path = "temp.jpg"
        photo.save(image_path)
        getSquares(image_path)
        
        generate = st.button('Genrate Prediction')
        if generate:
            
            total = 0 
            for i in three_notes:

                # Resize the image to 100x100
                resized = cv2.resize(np.array(i), (100, 100))
                resized = mobilenet_v2_preprocess_input(resized)
                img_reshape = resized[np.newaxis, ...]

                prediction = model.predict(img_reshape)
                

                if prediction >= 0.5:

                     total += 5000
                     st.write(5000)
                else:

                     total += 10000
                     st.write(10000)
#             st.success(total)
                   
        
            value = "The total Amount is",total,"Kyat"
            st.success(value)


            
elif option == ("Money Classification With White Background"):
    st.markdown("<h3 style='color: #1A5F7A;'>Money Classification With White Background</h3>", unsafe_allow_html= True)
    st.write("* This only predicts a bank note on a white background")
    st.write("* With Messy Background, the prediction will be wrong")

    #load file
    uploaded_file = st.file_uploader("Upload Your Image",type=["png", "jpg"])

    if uploaded_file is None:
        st.success("Please upload your image")

    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image Input')


        # Resize the image to 100x100
        resized = cv2.resize(np.array(image), (100, 100))
        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis, ...]

        generate = st.button('Genrate Prediction')
        if generate:
            prediction = model.predict(img_reshape)
            if prediction >= 0.5:
                text = "It is 5000 kyat"
                st.header(text)
            else:
                text = "It is 10000 kyat"
                st.header(text)
