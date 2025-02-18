import cv2
import mediapipe as mp
import math

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Função para calcular o ângulo entre três pontos
def calculate_angle(a, b, c):
    # Converter pontos para vetores
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    # Calcular o produto escalar e as magnitudes
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    # Calcular o ângulo em radianos e converter para graus
    angle = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle)

# Capturar vídeo da câmera do Mac
cap = cv2.VideoCapture(1)  # O número 0 indica a câmera padrão (webcam integrada)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break
    
    # Converter a imagem para RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processar a imagem com MediaPipe Pose
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Desenhar os pontos-chave e conexões na imagem
        mp_drawing.draw_landmarks(
            frame,  # Imagem de saída
            results.pose_landmarks,  # Pontos-chave detectados
            mp_pose.POSE_CONNECTIONS  # Conexões entre os pontos
        )
        
        # Extrair coordenadas dos pontos-chave
        landmarks = results.pose_landmarks.landmark
        
        # Exemplo: Calcular ângulo do cotovelo esquerdo
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        angle_elbow_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
        
        # Exibir o ângulo na tela
        cv2.putText(frame, f"Angulo Cotovelo: {angle_elbow_left:.2f} graus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Exibir a imagem
    cv2.imshow('Pose Detection - Mac Camera', frame)
    
    # Parar ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()