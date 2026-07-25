import cv2
import mediapipe as mp
import os
import glob
import math

# ==========================================
# 1. CARREGAMENTO DA IMAGEM
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
search_pattern = os.path.join(current_dir, "naogrita.*")
files_found = glob.glob(search_pattern)

if not files_found:
    print(f"ERRO: Nenhuma imagem 'naogrita' encontrada na pasta: {current_dir}")
    exit()

OVERLAY_IMAGE_PATH = files_found[0]
overlay_img = cv2.imread(OVERLAY_IMAGE_PATH)

if overlay_img is None:
    print("ERRO: Falha ao carregar a imagem. O arquivo pode estar corrompido.")
    exit()

print("Imagem carregada com sucesso e pronta para uso!")

# ==========================================
# 2. CONFIGURAÇÃO DO MEDIAPIPE (Rosto e Mãos)
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# 3. INICIALIZAÇÃO DA CÂMERA
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    exit()

print("Pressione a tecla 'q' na janela do vídeo para encerrar.")

# Inicializa o modelo Holistic (detecta corpo, rosto e mãos)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Efeito espelho
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # O MediaPipe precisa da imagem em RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        silence_detected = False

        # ==========================================
        # 4. LÓGICA DE DETECÇÃO (Mão na Boca)
        # ==========================================
        if results.face_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            # Ponto 13 é o centro do lábio superior
            mouth_landmark = results.face_landmarks.landmark[13]
            mouth_x, mouth_y = int(mouth_landmark.x * w), int(mouth_landmark.y * h)

            def check_hand_near_mouth(hand_landmarks):
                # Ponto 8 é a ponta do dedo indicador
                index_finger = hand_landmarks.landmark[8]
                finger_x, finger_y = int(index_finger.x * w), int(index_finger.y * h)
                
                # Calcula a distância entre o dedo e a boca
                distancia = math.hypot(mouth_x - finger_x, mouth_y - finger_y)
                
                # Se a distância for menor que 60 pixels, consideramos "mão na boca"
                return distancia < 60

            # Verifica a mão direita
            if results.right_hand_landmarks:
                if check_hand_near_mouth(results.right_hand_landmarks):
                    silence_detected = True
            
            # Verifica a mão esquerda
            if results.left_hand_landmarks and not silence_detected:
                if check_hand_near_mouth(results.left_hand_landmarks):
                    silence_detected = True

        # ==========================================
        # 5. EXIBIÇÃO NA TELA
        # ==========================================
        if silence_detected:
            # Exibe a foto do Cristiano Ronaldo
            overlay_resized = cv2.resize(overlay_img, (w, h))
            cv2.imshow("Mão na Boca - CR7", overlay_resized)
        else:
            # Desenha as marcações do rosto e mãos para você ver a detecção funcionando
            if results.face_landmarks:
                mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.imshow("Mão na Boca - CR7", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Limpeza
cap.release()
cv2.destroyAllWindows()