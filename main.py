# This script installs the required packages for the project.
import subprocess
import os

# Install other required packages
required_packages = [
    'tensorflow==2.16.1',
    'opencv-python',
    'mediapipe',
    'scikit-learn',
    'matplotlib'
]

import cv2
import numpy as np
import mediapipe as mp

############################################ KEYPOINTS USING MEDIAPIPE MODELS #############################################


mp_holistic = mp.solutions.holistic # Holistic model, makes detection easier
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_pose = mp.solutions.pose # Pose model
mp_face_mesh = mp.solutions.face_mesh # Face mesh model

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image,results):
     # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                   mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)#Giving a diferent color to the left hand, with is the hand that plays the guitar 
                                    )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

cap = cv2.VideoCapture(0) # Access the webcam
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read() #return status variable and the frame 

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Print pose landmarks for debugging
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                print(f"Pose landmark {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release() # Turn off the webcam
cv2.destroyAllWindows() # Closes the window

############################################ EXTRACT KEYPOINT VALUES #############################################

def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility]for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x,res.y,res.z]for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3) #Error Handling, the correct number of landmarks is 21       
    lhand = np.array([[res.x,res.y,res.z]for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) #Error Handling, the correct number of landmarks is 21       
    rhand = np.array([[res.x,res.y,res.z]for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) #Error Handling, the correct number of landmarks is 21
    
    return np.concatenate([pose,face,lhand,rhand])#Concatenating all the keypoints in one array

#print(extract_keypoints(results).shape) #Test the function

############################################ SETUP FOLDERS FOR COLLECTION #############################################

#Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') #Creating a folder to store the data

#Actions that we try to detect
actions = np.array(['C','D','E','F','G','A','B']) #The notes that will be played [ADD MORE NOTES IF NEEDED]

num_sequences = 30 #Number of sequences that will be recorded (30)
sequence_length = 30 #Number of frames that will be recorded (30 frames per sequence)

