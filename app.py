#import Libraries
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import imutils
# from imutils.perspective import four_point_transform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input



# Set page config
st.set_page_config(
    page_title = "Fake Momeny",
    page_icon = "img/dollar.jpg",
    layout="centered", 
    initial_sidebar_state="expanded"
)



#<---- functions start ---->

# loaded the model from other path 
@st.cache
def loadModel():
    model = tf.keras.models.load_model("v1.h5",compile= False)
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
    # img_copy1 = img.copy() # for display purposes

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
    
    
    for i in range(0,len(rect_cnts),2):
        document = four_point_transform(img_copy, rect_cnts[i].reshape(4, 2))
        three_notes.append(document)
        #   print(type(document), document.shape,  'type of doc')
        # show_images(['image', 'document'],  [document])
        st.image(document)




# <--- factions end --->

# A5D7E8

#<--- slide starting --->

st.sidebar.image("img/dollar.jpg")

# Options menu
option = st.sidebar.selectbox("Chosse what you want to do",
                            ("Limitations","Money Classification With Color Background","Money Classification With White Background"))

#<--- slide ending -->

st.markdown("<h1 style='color: #19376D;'>Classification Of Myanmar Bank Note</h1>", unsafe_allow_html= True)

if option == "Limitations":
    # st.subheader("Limitation")
        st.markdown("<h3 style='color: #1A5F7A;'>Limitation</h3>", unsafe_allow_html= True)
        st.write("- We can only classify five or ten thousand kyats")
        st.write("- We are able to extract 3 notes of currency from one input image")
        st.write("- Accuracy depends on the background of the image, more clear backgrounds make our model detect currency notes more precisely. ")
        st.write("- White backgrounds make the model hard to extract the currency ")
        st.write("- We will be able to calculate the total amount of money in the image and read it out loud for blind people")



elif option == "Money Classification With Color Background":
    st.markdown("<h3 style='color: #1A5F7A;'>Money Classification With Color Background</h3>", unsafe_allow_html= True)

    #load file
    uploaded_file = st.file_uploader("Upload Your Image",type=["png", "jpg"])

    if uploaded_file is None:
        st.success("Please upload your image")
    
    else:
        st.success("Your image has been uploaded")
        photo = Image.open(uploaded_file)
        st.image(photo, caption='Your image has been uploaded') 
        # getSquares(image)
        image_path = "temp.jpg"
        photo.save(image_path)
        getSquares(image_path)






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

        genrate = st.button('Genrate Prediction')
        if genrate:
            prediction = model.predict(img_reshape)
            if prediction >= 0.5:
                st.header("It is 5000 kyat")
            else:
                st.header("It is 10000 kyat")


       










# three_notes = []

# def show_images(titles , images, wait=True):
#     """Display multiple images with one line of code"""

#     for (title, image) in zip(titles, images): # why is zip 
#         plt.title(title)
#         plt.imshow(image)
#         plt.axis("off")
#         plt.show()

# def getSquares(input):
#     height = 600
#     width = 800
#     green = (0, 255, 0) # green color
#     red = (0, 0, 255) # red color
#     white = (255, 255, 255) # white color
#     # questions = 5
#     # answers = 5
#     # correct_ans = [0, 2, 1, 3, 4]

#     img = cv2.imread(ft)
#     img = cv2.resize(img, (width, height))
#     img_copy = img.copy() # for display purposes
#     # img_copy1 = img.copy() # for display purposes

#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
#     edge_img = cv2.Canny(blur_img, 20, 180) # adjust with 


#     # find the contours in the image
#     contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
#     # get rect cnts
#     rect_cnts = []
#     for cnt in contours:
#         # approximate the contour
#         peri = cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#         # if the approximated contour is a rectangle ...
#         if len(approx) == 4:
#             # append it to our list
#             rect_cnts.append(approx)
#     # sort the contours from biggest to smallest
#     rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)
    
    
#     for i in range(0,6,2):
#         document = four_point_transform(img_copy, rect_cnts[i].reshape(4, 2))
#         three_notes.append(document)
#         #   print(type(document), document.shape,  'type of doc')
#         show_images(['image', 'document'],  [document])