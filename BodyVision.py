import cv2
import mediapipe as mp
import math
import numpy as np

class PoseDetector:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode,
                                      min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    @staticmethod
    def calculate_angle(a, b, c):
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        
        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0
        
        angle_radians = math.acos(dot_product / (magnitude_ba * magnitude_bc))
        return math.degrees(angle_radians)

    @staticmethod
    def evaluate_double_biceps(left_angle, right_angle, left_elbow_height, right_elbow_height, left_shoulder_height, right_shoulder_height):
        if left_elbow_height > left_shoulder_height or right_elbow_height > right_shoulder_height:
            return "Posicao incorreta - Levante os cotovelos na altura dos ombros."
        if left_elbow_height < left_shoulder_height * 0.7 or right_elbow_height < right_shoulder_height * 0.7:
            return "Posicao incorreta - Bracos muito abaixados."
        if left_angle > 90 or right_angle > 90:
            return "Posicao incorreta - Ajuste os braços para formar um angulo menor ou igual a 90 graus."
        return "Posicao correta - Excelente postura!"

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            
            shoulder_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                          landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                          landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            shoulder_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            angle_elbow_left = self.calculate_angle(shoulder_left, elbow_left, wrist_left)
            angle_elbow_right = self.calculate_angle(shoulder_right, elbow_right, wrist_right)
            
            left_elbow_height = elbow_left[1]
            right_elbow_height = elbow_right[1]
            left_shoulder_height = shoulder_left[1]
            right_shoulder_height = shoulder_right[1]
            
            pose_quality = self.evaluate_double_biceps(angle_elbow_left, angle_elbow_right, left_elbow_height, right_elbow_height, left_shoulder_height, right_shoulder_height)
            
            text_color = (0, 255, 0) if "Posicaoo correta" in pose_quality else (0, 0, 255)
            angle_text_color = (139, 0, 0)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8  
            thickness = 3
            
            ##cv2.putText(frame, f"Angulo Cotovelo Esquerdo: {angle_elbow_left:.2f} graus", (10, 30), font, font_scale, angle_text_color, thickness)
            ##cv2.putText(frame, f"Angulo Cotovelo Direito: {angle_elbow_right:.2f} graus", (10, 60), font, font_scale, angle_text_color, thickness)
            cv2.putText(frame, f"Qualidade: {pose_quality}", (10, 30), font, font_scale, text_color, thickness)
        
        return frame

def main():
    detector = PoseDetector()
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erro ao acessar a camera.")
        return
    
    # Criando uma janela em tela cheia
    cv2.namedWindow('Pose Detection - Mac Camera', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pose Detection - Mac Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break
        
        frame = detector.process_frame(frame)
        cv2.imshow('Pose Detection - Mac Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
