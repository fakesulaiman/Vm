import imageio
import mediapipe as mp
import pyautogui
import numpy as np
from PIL import Image

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture using imageio
video = imageio.get_reader('<video0>', 'ffmpeg')

# Screen size for controlling the mouse
screen_width, screen_height = pyautogui.size()

def show_image(img):
    img_pil = Image.fromarray(img)
    img_pil.show()

try:
    for frame in video:
        # Convert frame to RGB
        frame_rgb = np.array(frame)
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        result = hands.process(frame_rgb)

        # Draw hand landmarks and calculate finger tips
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Get the coordinates of the index finger tip
                index_finger_tip = hand_landmarks.landmark[8]
                # Convert the normalized coordinates to screen coordinates
                x = int(index_finger_tip.x * screen_width)
                y = int(index_finger_tip.y * screen_height)
                # Move the mouse to the new position
                pyautogui.moveTo(x, y)

        # Convert the frame back to BGR for displaying with imageio
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        show_image(frame_bgr)

        # Add a small delay to prevent high CPU usage
        pyautogui.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    video.close()
