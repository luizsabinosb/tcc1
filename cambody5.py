import cv2
import mediapipe as mp
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Função para calcular o ângulo entre três pontos
def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0  # Evitar divisão por zero
    
    angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle)

# Função para avaliar a qualidade do duplo bíceps
def evaluate_double_biceps(left_angle, right_angle, left_elbow_height, right_elbow_height, left_shoulder_height, right_shoulder_height):
    min_elbow_height = left_shoulder_height * 0.9  # Limite mínimo para a altura relativa dos cotovelos
    max_elbow_height = left_shoulder_height * 1.1  # Limite máximo
    
    if left_elbow_height > left_shoulder_height or right_elbow_height > right_shoulder_height:
        return "Posicao incorreta - Levante os cotovelos na altura dos ombros."
    if left_elbow_height < left_shoulder_height * 0.7 or right_elbow_height < right_shoulder_height * 0.7:
        return "Posicao incorreta - Bracos muito abaixados."
    if left_angle < 150 or right_angle < 150:
        return "Posicao incorreta - Ajuste os bracos para formar um angulo proximo de 150 graus."
    return "Posicao correta - Excelente postura!"

# Capturar vídeo da câmera do Mac
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a camera.")
        break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        landmarks = results.pose_landmarks.landmark
        
        # Obter coordenadas dos pontos-chave
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calcular ângulos dos cotovelos
        angle_elbow_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
        angle_elbow_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
        
        # Obter alturas dos cotovelos e ombros
        left_elbow_height = elbow_left[1]
        right_elbow_height = elbow_right[1]
        left_shoulder_height = shoulder_left[1]
        right_shoulder_height = shoulder_right[1]
        
        pose_quality = evaluate_double_biceps(angle_elbow_left, angle_elbow_right, left_elbow_height, right_elbow_height, left_shoulder_height, right_shoulder_height)
        
        # Definir a cor do texto com base na qualidade da pose
        if "Posição correta" in pose_quality:
            text_color = (0, 255, 0)  # Verde
        else:
            text_color = (0, 0, 255)  # Vermelho
        
        # Alterar a cor do texto para azul escuro para as mensagens de ângulo
        angle_text_color = (139, 0, 0)  # Azul escuro (dark blue)

        # Converter imagem de OpenCV para Pillow (PIL)
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Carregar uma fonte bonita
        try:
            font = ImageFont.truetype("arial.ttf", 30)  # Fonte Arial (substitua se necessário)
        except IOError:
            font = ImageFont.load_default()  # Fonte padrão caso a fonte não seja encontrada
        
        # Adicionar o texto com a fonte personalizada
        draw.text((10, 30), f"Angulo Esq: {angle_elbow_left:.2f} graus", font=font, fill=angle_text_color)
        draw.text((10, 70), f"Angulo Dir: {angle_elbow_right:.2f} graus", font=font, fill=angle_text_color)
        draw.text((10, 110), f"Qualidade: {pose_quality}", font=font, fill=text_color)
        
        # Converter de volta para OpenCV
        frame = np.array(pil_image)
    
    cv2.imshow('Pose Detection - Mac Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
