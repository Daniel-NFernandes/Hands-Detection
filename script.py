import cv2
import mediapipe as mp

draw = mp.solutions.drawing_utils
hands_mp = mp.solutions.hands
hands = hands_mp.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

# Definindo a resolução da janela
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    sucesso, frame = cap.read()

    if sucesso:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(frameRGB)

        if resultado.multi_hand_landmarks:
            landmarks = []  # Lista para armazenar os pontos de referência das mãos
            for lm in resultado.multi_hand_landmarks:
                draw.draw_landmarks(frame, lm, hands_mp.HAND_CONNECTIONS)
                # Coletar todos os pontos de referência das mãos
                landmarks.append(lm.landmark)
                print(lm)

            if len(landmarks) == 2:
                # Desenhar linhas entre os pontos correspondentes de ambas as mãos
                for i in range(len(landmarks[0])):
                    x1, y1 = int(landmarks[0][i].x * frame.shape[1]), int(landmarks[0][i].y * frame.shape[0])
                    x2, y2 = int(landmarks[1][i].x * frame.shape[1]), int(landmarks[1][i].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("Video", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()