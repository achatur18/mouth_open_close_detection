import cv2
import streamlit as st
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def l2_dist(p1, p2):
    result=(((p1.x-p2.x)**2)+((p1.y-p2.y)**2))**0.5
    return result

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def get_mesh_image(image):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            dist=l2_dist(face_landmarks.landmark[13], face_landmarks.landmark[14])
            if int(dist*100):
                image = cv2.putText(image, 'Open Mouth', org, font, 
                                fontScale, color, thickness, cv2.LINE_AA)
            else:
                image = cv2.putText(image, 'Close Mouth', org, font, 
                                fontScale, color, thickness, cv2.LINE_AA)
    return image
            

st.title("Mouth Open/Close Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(get_mesh_image(frame))
else:
    st.write('Stopped')