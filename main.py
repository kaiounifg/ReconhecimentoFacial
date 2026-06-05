import face_recognition
import cv2
import numpy as np
import os
import sys
import threading
import time

class SistemaAcessoUlife:
    def __init__(self):
        self.diretorio_fotos = "cadastrados"
        self.arquivo_encodings = "data/encodings_ulife.pickle"
        self.conhecidos_encodings = []
        self.conhecidos_ras = []
        
        self.frame_atual = None
        self.face_locations = []
        self.face_names = []
        self.face_colors = []
        self.running = True
        self.ready_to_process = True 
        
        # Variáveis de Interface
        self.modo_input = None # Pode ser 'cadastro' ou 'deletar'
        self.nome_digitado = ""
        self.frame_capturado = None

        if not os.path.exists(self.diretorio_fotos): os.makedirs(self.diretorio_fotos)
        if not os.path.exists("data"): os.makedirs("data")

    def treinar_sistema(self):
        """Atualiza os rostos conhecidos na memória"""
        novos_encodings, novos_ras = [], []
        arquivos = [f for f in os.listdir(self.diretorio_fotos) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        for arquivo in arquivos:
            caminho = os.path.join(self.diretorio_fotos, arquivo)
            ra = os.path.splitext(arquivo)[0]
            img = face_recognition.load_image_file(caminho)
            encodings = face_recognition.face_encodings(img)
            if len(encodings) > 0:
                novos_encodings.append(encodings[0])
                novos_ras.append(ra)
        
        self.conhecidos_encodings = novos_encodings
        self.conhecidos_ras = novos_ras

    def _worker_reconhecimento(self):
        while self.running:
            if self.frame_atual is not None and self.ready_to_process and not self.modo_input:
                self.ready_to_process = False 
                rgb_frame = self.frame_atual.copy()
                locations = face_recognition.face_locations(rgb_frame, model="hog")
                encodings = face_recognition.face_encodings(rgb_frame, locations)
                temp_names, temp_colors = [], []

                for face_encoding in encodings:
                    ra_exibido, cor = "DESCONHECIDO", (0, 0, 255)
                    if len(self.conhecidos_encodings) > 0:
                        matches = face_recognition.compare_faces(self.conhecidos_encodings, face_encoding, tolerance=0.5)
                        dist = face_recognition.face_distance(self.conhecidos_encodings, face_encoding)
                        if len(dist) > 0 and matches[np.argmin(dist)]:
                            ra_exibido = f"RA: {self.conhecidos_ras[np.argmin(dist)]}"
                            cor = (0, 255, 0)
                    temp_names.append(ra_exibido)
                    temp_colors.append(cor)

                self.face_locations, self.face_names, self.face_colors = locations, temp_names, temp_colors
                self.ready_to_process = True 
            else:
                time.sleep(0.01)

    def iniciar_reconhecimento(self):
        video_capture = cv2.VideoCapture(0)
        nome_janela = 'Ulife FaceID'
        cv2.namedWindow(nome_janela, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        threading.Thread(target=self._worker_reconhecimento, daemon=True).start()

        while True:
            ret, frame = video_capture.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            if not self.modo_input:
                # MODO NORMAL
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                self.frame_atual = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                for (top, right, bottom, left), name, cor in zip(self.face_locations, self.face_names, self.face_colors):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)
                    cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor, 1)
                
                cv2.putText(frame, "C: Cadastrar | D: Deletar | Q: Sair", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            else:
                # MODO DE DIGITAÇÃO (CADASTRO OU DELEÇÃO)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 160), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                titulo = "NOVO CADASTRO" if self.modo_input == 'cadastro' else "DELETAR CADASTRO (RA/NOME)"
                cor_texto = (0, 255, 0) if self.modo_input == 'cadastro' else (0, 0, 255)
                
                cv2.putText(frame, titulo, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, f"NOME: {self.nome_digitado}", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1.2, cor_texto, 2)
                cv2.putText(frame, "ENTER: Confirmar | ESC: Cancelar", (50, 145), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(nome_janela, frame)
            key = cv2.waitKey(1)

            if not self.modo_input:
                if key == ord('q') or key == ord('Q'): break
                elif key == ord('c') or key == ord('C'):
                    self.modo_input = 'cadastro'
                    self.nome_digitado = ""
                    self.frame_capturado = frame.copy()
                elif key == ord('d') or key == ord('D'):
                    self.modo_input = 'deletar'
                    self.nome_digitado = ""
            else:
                if key == 27: # ESC
                    self.modo_input = None
                elif key == 13: # ENTER
                    nome_alvo = self.nome_digitado.strip()
                    if self.modo_input == 'cadastro' and nome_alvo:
                        cv2.imwrite(os.path.join(self.diretorio_fotos, f"{nome_alvo}.jpg"), self.frame_capturado)
                    elif self.modo_input == 'deletar' and nome_alvo:
                        caminho = os.path.join(self.diretorio_fotos, f"{nome_alvo}.jpg")
                        if os.path.exists(caminho): os.remove(caminho)
                    
                    self.treinar_sistema()
                    self.modo_input = None
                elif key == 8: # Backspace
                    self.nome_digitado = self.nome_digitado[:-1]
                elif 32 <= key <= 126:
                    self.nome_digitado += chr(key)

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SistemaAcessoUlife()
    app.treinar_sistema()
    app.iniciar_reconhecimento()
