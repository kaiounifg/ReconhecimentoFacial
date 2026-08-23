import face_recognition
import cv2
import numpy as np
import os
import threading
import time

class SistemaAcessoUlife:
    def __init__(self):
        self.diretorio_fotos = "cadastrados"
        self.conhecidos_encodings = []
        self.conhecidos_ras = []
        
        self.frame_atual = None
        self.face_locations = []
        self.face_names = []
        self.face_colors = []
        self.face_confiancas = []
        self.instrucoes_distancia = []
        
        # Controle de temporizador de 2.5 segundos para liberação
        self.tempo_reconhecimento_inicio = {}
        self.progresso_reconhecimento = {}
        self.TEMPO_NECESSARIO = 2.5
        
        self.running = True
        self.ready_to_process = True 
        
        # Variáveis de Interface
        self.modo_input = None # 'cadastro' ou 'deletar'
        self.nome_digitado = ""
        self.frame_capturado = None
        
        self.ultimo_acesso_nome = ""
        self.tempo_ultimo_acesso = 0

        if not os.path.exists(self.diretorio_fotos): os.makedirs(self.diretorio_fotos)
        if not os.path.exists("data"): os.makedirs("data")

    def treinar_sistema(self):
        novos_encodings, novos_ras = [], []
        if not os.path.exists(self.diretorio_fotos): return

        arquivos = [f for f in os.listdir(self.diretorio_fotos) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        for arquivo in arquivos:
            caminho = os.path.join(self.diretorio_fotos, arquivo)
            ra = os.path.splitext(arquivo)[0]
            try:
                img = face_recognition.load_image_file(caminho)
                encodings = face_recognition.face_encodings(img)
                if len(encodings) > 0:
                    novos_encodings.append(encodings[0])
                    novos_ras.append(ra)
            except Exception as e:
                print(f"[AVISO] Erro ao carregar {arquivo}: {e}")
        
        self.conhecidos_encodings = novos_encodings
        self.conhecidos_ras = novos_ras

    def _worker_reconhecimento(self):
        while self.running:
            if self.frame_atual is not None and self.ready_to_process and not self.modo_input:
                self.ready_to_process = False 
                rgb_frame = self.frame_atual.copy()
                
                locations = face_recognition.face_locations(rgb_frame, model="hog")
                encodings = face_recognition.face_encodings(rgb_frame, locations)
                
                temp_names, temp_colors, temp_confs, temp_dist = [], [], [], []
                rostos_detectados_agora = set()

                for (top, right, bottom, left), face_encoding in zip(locations, encodings):
                    ra_exibido, cor, conf_str = "DESCONHECIDO", (180, 180, 180), "" # Cinza claro para desconhecido
                    
                    # Verificação de distância / tamanho do rosto
                    altura_face = bottom - top
                    if altura_face < 35:
                        status_dist, cor_dist = "APROXIME O ROSTO", (0, 165, 255)
                    elif altura_face > 110:
                        status_dist, cor_dist = "AFASTE O ROSTO", (0, 165, 255)
                    else:
                        status_dist, cor_dist = "POSICAO IDEAL", (0, 255, 120)

                    if len(self.conhecidos_encodings) > 0:
                        matches = face_recognition.compare_faces(self.conhecidos_encodings, face_encoding, tolerance=0.48)
                        dist = face_recognition.face_distance(self.conhecidos_encodings, face_encoding)
                        
                        if len(dist) > 0:
                            best_match_index = np.argmin(dist)
                            if matches[best_match_index]:
                                nome_identificado = self.conhecidos_ras[best_match_index]
                                ra_exibido = f"ID: {nome_identificado}"
                                cor = (255, 255, 255) # Branco elegante para reconhecido
                                confianca = max(0, min(100, (1 - dist[best_match_index]) * 100))
                                conf_str = f"{confianca:.1f}%"
                                
                                rostos_detectados_agora.add(nome_identificado)
                                
                                # Lógica de contagem de 2.5 segundos
                                tempo_atual = time.time()
                                if nome_identificado not in self.tempo_reconhecimento_inicio:
                                    self.tempo_reconhecimento_inicio[nome_identificado] = tempo_atual
                                
                                decorrido = tempo_atual - self.tempo_reconhecimento_inicio[nome_identificado]
                                progresso = min(1.0, decorrido / self.TEMPO_NECESSARIO)
                                self.progresso_reconhecimento[nome_identificado] = progresso
                                
                                if progresso >= 1.0:
                                    self.ultimo_acesso_nome = nome_identificado
                                    self.tempo_ultimo_acesso = tempo_atual

                    temp_names.append(ra_exibido)
                    temp_colors.append(cor)
                    temp_confs.append(conf_str)
                    temp_dist.append((status_dist, cor_dist))

                for nome in list(self.tempo_reconhecimento_inicio.keys()):
                    if nome not in rostos_detectados_agora:
                        del self.tempo_reconhecimento_inicio[nome]
                        if nome in self.progresso_reconhecimento:
                            del self.progresso_reconhecimento[nome]

                self.face_locations = locations
                self.face_names = temp_names
                self.face_colors = temp_colors
                self.face_confiancas = temp_confs
                self.instrucoes_distancia = temp_dist
                self.ready_to_process = True 
            else:
                time.sleep(0.01)

    def desenhar_cantos_retangulo(self, img, pt1, pt2, color, thickness=2, r_len=18):
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.line(img, (x1, y1), (x1 + r_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + r_len), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - r_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + r_len), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + r_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - r_len), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - r_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - r_len), color, thickness, cv2.LINE_AA)

    def desenhar_painel_superior(self, frame):
        altura, largura = frame.shape[:2]
        overlay = frame.copy()
        
        # Cabeçalho limpo sem expor comandos (Segurança para Totem)
        cv2.rectangle(overlay, (0, 0), (largura, 65), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, "ULIFE ACCESS // CONTROLE DE CATRACA", (30, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Posicione o rosto no visor", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    def iniciar_reconhecimento(self):
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not video_capture.isOpened():
            print("[ERRO CRÍTICO] Webcam inacessível.")
            return

        nome_janela = 'Ulife Totem Secure'
        cv2.namedWindow(nome_janela, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        threading.Thread(target=self._worker_reconhecimento, daemon=True).start()

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                altura, largura = frame.shape[:2]

                if not self.modo_input:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    self.frame_atual = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    for (top, right, bottom, left), name, cor, conf, (status_dist, cor_dist) in zip(
                        self.face_locations, self.face_names, self.face_colors, self.face_confiancas, self.instrucoes_distancia
                    ):
                        top *= 4; right *= 4; bottom *= 4; left *= 4
                        
                        # Desenha moldura branca minimalista
                        self.desenhar_cantos_retangulo(frame, (left, top), (right, bottom), cor, thickness=2, r_len=18)
                        
                        # Caixa de identificação inferior
                        label_bg_bottom = bottom + 34
                        cv2.rectangle(frame, (left, bottom), (right, label_bg_bottom), (35, 35, 35), -1)
                        cv2.rectangle(frame, (left, bottom), (right, label_bg_bottom), cor, 1)
                        cv2.putText(frame, name, (left + 8, bottom + 23), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        # Barra de progresso dos 2.5 segundos em branco/verde
                        if "ID: " in name:
                            nome_chave = name.replace("ID: ", "").strip()
                            if nome_chave in self.progresso_reconhecimento:
                                prog = self.progresso_reconhecimento[nome_chave]
                                largura_barra = right - left
                                altura_barra = 5
                                barra_preenchida = int(largura_barra * prog)
                                
                                cv2.rectangle(frame, (left, label_bg_bottom + 4), (right, label_bg_bottom + 4 + altura_barra), (60, 60, 60), -1)
                                if barra_preenchida > 0:
                                    cv2.rectangle(frame, (left, label_bg_bottom + 4), (left + barra_preenchida, label_bg_bottom + 4 + altura_barra), (255, 255, 255), -1)

                        if conf:
                            cv2.putText(frame, conf, (right - 55, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                        # Guia de distância flutuante
                        cv2.putText(frame, status_dist, (left, top - 12), cv2.FONT_HERSHEY_DUPLEX, 0.45, cor_dist, 1, cv2.LINE_AA)

                    # Banner centralizado de acesso liberado (Catraca aberta)
                    if time.time() - self.tempo_ultimo_acesso < 3.0 and self.ultimo_acesso_nome:
                        banner_w, banner_h = 420, 80
                        bx1 = (largura - banner_w) // 2
                        by1 = 90
                        bx2 = bx1 + banner_w
                        by2 = by1 + banner_h
                        
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (25, 25, 25), -1)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
                        cv2.putText(frame, "ACESSO LIBERADO", (bx1 + 105, by1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Bem-vindo(a), {self.ultimo_acesso_nome}", (bx1 + 65, by1 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

                    self.desenhar_painel_superior(frame)
                
                else:
                    # Janela Modal oculta do usuário comum (Ativada via teclado pelo admin)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (largura, altura), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                    
                    modal_l, modal_a = 640, 240
                    x1 = (largura - modal_l) // 2
                    y1 = (altura - modal_a) // 2
                    x2 = x1 + modal_l
                    y2 = y1 + modal_a
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
                    is_cadastro = (self.modo_input == 'cadastro')
                    titulo = "MODO ADMIN: CADASTRAR ROSTO" if is_cadastro else "MODO ADMIN: DELETAR REGISTRO"
                    cor_texto = (255, 255, 255) if is_cadastro else (0, 0, 255)
                    
                    cv2.putText(frame, titulo, (x1 + 30, y1 + 42), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"RA/NOME: {self.nome_digitado}_", (x1 + 30, y1 + 120), cv2.FONT_HERSHEY_DUPLEX, 0.85, cor_texto, 2, cv2.LINE_AA)
                    cv2.putText(frame, "[ENTER] Confirmar    |    [ESC] Sair do Modo Admin", (x1 + 30, y1 + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

                cv2.imshow(nome_janela, frame)
                key = cv2.waitKey(1) & 0xFF

                if not self.modo_input:
                    if key == ord('q') or key == ord('Q'): 
                        break
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
                        if is_cadastro and nome_alvo:
                            caminho_salvar = os.path.join(self.diretorio_fotos, f"{nome_alvo}.jpg")
                            cv2.imwrite(caminho_salvar, self.frame_capturado)
                        elif not is_cadastro and nome_alvo:
                            caminho = os.path.join(self.diretorio_fotos, f"{nome_alvo}.jpg")
                            if os.path.exists(caminho): 
                                os.remove(caminho)
                        
                        self.treinar_sistema()
                        self.modo_input = None
                    elif key == 8 or key == 127: # Backspace
                        self.nome_digitado = self.nome_digitado[:-1]
                    elif 32 <= key <= 126:
                        self.nome_digitado += chr(key)
        finally:
            self.running = False
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SistemaAcessoUlife()
    app.treinar_sistema()
    app.iniciar_reconhecimento()
