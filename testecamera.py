import cv2

print("Tentando abrir a câmera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não consegui acessar a webcam.")
else:
    print("Câmera aberta! Aperte 'Q' para fechar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Teste', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()