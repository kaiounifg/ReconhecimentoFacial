import os
import threading
import time
import cv2
import face_recognition
import numpy as np
import winsound

# Import opcional para comunicação serial (catraca física)
try:
  import serial
  import serial.tools.list_ports

  SERIAL_DISPONIVEL = True
except ImportError:
  SERIAL_DISPONIVEL = False


class SistemaAcessoUlife:

  def __init__(self):
    self.diretorio_fotos = "cadastrados"
    self.conhecidos_encodings = []
    self.conhecidos_ras = []

    self.frame_atual = None
    self.frame_limpo_atual = None
    self.face_locations = []
    self.face_names = []
    self.face_colors = []
    self.face_confiancas = []
    self.instrucoes_distancia = []

    self.tempo_reconhecimento_inicio = {}
    self.progresso_reconhecimento = {}
    self.TEMPO_NECESSARIO = 2.5

    self.acesso_liberado_recente = False
    self.acesso_negado_recente = False

    self.running = True
    self.ready_to_process = True

    self.modo_input = None
    self.nome_digitado = ""
    self.frame_capturado = None

    self.ultimo_acesso_nome = ""
    self.tempo_ultimo_acesso = 0

    self.ultimo_negado_nome = ""
    self.tempo_ultimo_negado = 0

    # Configuração da Catraca Física (Serial / Arduino)
    self.porta_serial_nome = "COM3"
    self.baud_rate = 9600
    self.conexao_catraca = None
    self.inicializar_catraca_fisica()

    if not os.path.exists(self.diretorio_fotos):
      os.makedirs(self.diretorio_fotos)
    if not os.path.exists("data"):
      os.makedirs("data")

  def inicializar_catraca_fisica(self):
    if SERIAL_DISPONIVEL:
      try:
        self.conexao_catraca = serial.Serial(
            self.porta_serial_nome, self.baud_rate, timeout=1
        )
        time.sleep(2)
        print(
            f"[INFO] Catraca física conectada com sucesso na porta"
            f" {self.porta_serial_nome}."
        )
      except Exception as e:
        print(
            f"[AVISO] Catraca física não encontrada em"
            f" {self.porta_serial_nome}. Operando em MODO TESTE (Apenas"
            f" Simulação/Bipe)."
        )
        self.conexao_catraca = None
    else:
      print(
          "[AVISO] Biblioteca 'pyserial' não instalada. Operando em MODO TESTE."
      )

  def acionar_catraca_fisica(self):
    if self.conexao_catraca and self.conexao_catraca.is_open:
      try:
        self.conexao_catraca.write(b"ABRIR\n")
      except Exception as e:
        print(f"[ERRO] Falha ao enviar comando para a catraca física: {e}")

  def treinar_sistema(self):
    novos_encodings, novos_ras = [], []
    if not os.path.exists(self.diretorio_fotos):
      return

    arquivos = [
        f
        for f in os.listdir(self.diretorio_fotos)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

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
      if (
          self.frame_atual is not None
          and self.ready_to_process
          and not self.modo_input
      ):
        self.ready_to_process = False
        rgb_frame = self.frame_atual.copy()

        locations = face_recognition.face_locations(rgb_frame, model="hog")
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        temp_names, temp_colors, temp_confs, temp_dist = [], [], [], []
        rostos_detectados_agora = set()

        for (top, right, bottom, left), face_encoding in zip(
            locations, encodings
        ):
          ra_exibido, cor, conf_str = (
              "DESCONHECIDO",
              (200, 200, 200),
              "",
          )
          is_cadastrado = False
          nome_identificado = "DESCONHECIDO"

          altura_face = bottom - top

          if altura_face < 65:
            status_dist, cor_dist = "APROXIME O ROSTO", (0, 140, 255)
          elif altura_face > 240:
            status_dist, cor_dist = "AFASTE O ROSTO", (0, 140, 255)
          else:
            status_dist, cor_dist = "", (0, 0, 0)

          if len(self.conhecidos_encodings) > 0:
            matches = face_recognition.compare_faces(
                self.conhecidos_encodings, face_encoding, tolerance=0.48
            )
            dist = face_recognition.face_distance(
                self.conhecidos_encodings, face_encoding
            )

            if len(dist) > 0:
              best_match_index = np.argmin(dist)
              if matches[best_match_index]:
                nome_identificado = self.conhecidos_ras[best_match_index]
                ra_exibido = f"ID: {nome_identificado}"
                cor = (255, 255, 255)
                confianca = max(
                    0, min(100, (1 - dist[best_match_index]) * 100)
                )
                conf_str = f"{confianca:.1f}%"
                is_cadastrado = True

          chave_temporizador = (
              nome_identificado
              if is_cadastrado
              else f"DESCONHECIDO_{left}_{top}"
          )
          rostos_detectados_agora.add(chave_temporizador)

          tempo_atual = time.time()
          if chave_temporizador not in self.tempo_reconhecimento_inicio:
            self.tempo_reconhecimento_inicio[chave_temporizador] = tempo_atual

          decorrido = (
              tempo_atual - self.tempo_reconhecimento_inicio[chave_temporizador]
          )
          progresso = min(1.0, decorrido / self.TEMPO_NECESSARIO)
          self.progresso_reconhecimento[chave_temporizador] = progresso

          if progresso >= 1.0:
            if is_cadastrado:
              self.ultimo_acesso_nome = nome_identificado
              self.tempo_ultimo_acesso = tempo_atual

              if not self.acesso_liberado_recente:
                try:
                  winsound.Beep(2200, 250)
                except:
                  pass
                self.acionar_catraca_fisica()
                self.acesso_liberado_recente = True
            else:
              self.ultimo_negado_nome = "DESCONHECIDO"
              self.tempo_ultimo_negado = tempo_atual

              if not self.acesso_negado_recente:
                try:
                  winsound.Beep(600, 400)
                except:
                  pass
                self.acesso_negado_recente = True

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

  def desenhar_cantos_retangulo(
      self, img, pt1, pt2, color, thickness=2, r_len=18
  ):
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

    cv2.rectangle(overlay, (0, 0), (largura, 70), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    cv2.line(frame, (0, 70), (largura, 70), (45, 45, 45), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "ULIFE ACCESS // SECURE GATEWAY",
        (35, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Sistema de Identificacao Facial Biometrica",
        (35, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

  def iniciar_reconhecimento(self):
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not video_capture.isOpened():
      print("[ERRO CRÍTICO] Webcam inacessível.")
      return

    nome_janela = "Ulife Totem Secure"
    cv2.namedWindow(nome_janela, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        nome_janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    threading.Thread(
        target=self._worker_reconhecimento, daemon=True
    ).start()

    try:
      while True:
        ret, frame = video_capture.read()
        if not ret:
          break

        frame = cv2.flip(frame, 1)
        altura, largura = frame.shape[:2]

        # Cópia limpa do frame sem elementos gráficos (usada para o cadastro)
        frame_limpo = frame.copy()

        if not self.modo_input:
          small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
          self.frame_atual = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

          if len(self.tempo_reconhecimento_inicio) == 0:
            self.acesso_liberado_recente = False
            self.acesso_negado_recente = False

          for (
              (top, right, bottom, left),
              name,
              cor,
              conf,
              (status_dist, cor_dist),
          ) in zip(
              self.face_locations,
              self.face_names,
              self.face_colors,
              self.face_confiancas,
              self.instrucoes_distancia,
          ):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            self.desenhar_cantos_retangulo(
                frame, (left, top), (right, bottom), cor, thickness=2, r_len=18
            )

            label_bg_bottom = bottom + 38
            cv2.rectangle(
                frame,
                (left, bottom),
                (right, label_bg_bottom),
                (25, 25, 25),
                -1,
            )
            cv2.rectangle(
                frame, (left, bottom), (right, label_bg_bottom), cor, 1
            )
            cv2.putText(
                frame,
                name,
                (left + 10, bottom + 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            chave_prog = (
                name.replace("ID: ", "").strip()
                if "ID: " in name
                else f"DESCONHECIDO_{left//2}_{top//2}"
            )
            if chave_prog in self.progresso_reconhecimento:
              prog = self.progresso_reconhecimento[chave_prog]
              largura_barra = right - left
              altura_barra = 4
              barra_preenchida = int(largura_barra * prog)

              cv2.rectangle(
                  frame,
                  (left, label_bg_bottom + 2),
                  (right, label_bg_bottom + 2 + altura_barra),
                  (50, 50, 50),
                  -1,
              )
              if barra_preenchida > 0:
                cv2.rectangle(
                    frame,
                    (left, label_bg_bottom + 2),
                    (left + barra_preenchida, label_bg_bottom + 2 + altura_barra),
                    (255, 255, 255),
                    -1,
                )

            if conf:
              cv2.putText(
                  frame,
                  conf,
                  (right - 58, top - 12),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.42,
                  (220, 220, 220),
                  1,
                  cv2.LINE_AA,
              )

            if status_dist:
              cv2.putText(
                  frame,
                  status_dist,
                  (left, top - 12),
                  cv2.FONT_HERSHEY_DUPLEX,
                  0.45,
                  cor_dist,
                  1,
                  cv2.LINE_AA,
              )

          if (
              time.time() - self.tempo_ultimo_acesso < 5.0
              and self.ultimo_acesso_nome
          ):
            banner_w, banner_h = 460, 90
            bx1 = (largura - banner_w) // 2
            by1 = 95
            bx2 = bx1 + banner_w
            by2 = by1 + banner_h

            overlay_banner = frame.copy()
            cv2.rectangle(
                overlay_banner, (bx1, by1), (bx2, by2), (20, 20, 20), -1
            )
            cv2.addWeighted(overlay_banner, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 220, 120), 2)
            cv2.putText(
                frame,
                "ACESSO LIBERADO",
                (bx1 + 120, by1 + 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 255, 130),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Bem-vindo(a), {self.ultimo_acesso_nome}",
                (bx1 + 75, by1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

          if (
              time.time() - self.tempo_ultimo_negado < 5.0
              and self.ultimo_negado_nome
          ):
            banner_w, banner_h = 460, 90
            bx1 = (largura - banner_w) // 2
            by1 = 95
            bx2 = bx1 + banner_w
            by2 = by1 + banner_h

            overlay_banner = frame.copy()
            cv2.rectangle(
                overlay_banner, (bx1, by1), (bx2, by2), (15, 10, 10), -1
            )
            cv2.addWeighted(overlay_banner, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 240), 2)
            cv2.putText(
                frame,
                "ACESSO NEGADO",
                (bx1 + 130, by1 + 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Rosto nao cadastrado no sistema",
                (bx1 + 90, by1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

          self.desenhar_painel_superior(frame)

        else:
          overlay = frame.copy()
          cv2.rectangle(overlay, (0, 0), (largura, altura), (0, 0, 0), -1)
          cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

          modal_l, modal_a = 660, 260
          x1 = (largura - modal_l) // 2
          y1 = (altura - modal_a) // 2
          x2 = x1 + modal_l
          y2 = y1 + modal_a

          cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
          cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

          is_cadastro = self.modo_input == "cadastro"
          titulo = (
              "PAINEL ADMIN // CADASTRAR REGISTRO"
              if is_cadastro
              else "PAINEL ADMIN // REMOVER REGISTRO"
          )
          cor_texto = (255, 255, 255) if is_cadastro else (80, 80, 255)

          cv2.putText(
              frame,
              titulo,
              (x1 + 35, y1 + 45),
              cv2.FONT_HERSHEY_DUPLEX,
              0.58,
              (240, 240, 240),
              1,
              cv2.LINE_AA,
          )
          cv2.putText(
              frame,
              f"ID / NOME: {self.nome_digitado}_",
              (x1 + 35, y1 + 130),
              cv2.FONT_HERSHEY_DUPLEX,
              0.8,
              cor_texto,
              2,
              cv2.LINE_AA,
          )
          cv2.putText(
              frame,
              "[ENTER] Confirmar Operacao    |    [ESC] Retornar",
              (x1 + 35, y1 + 205),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.45,
              (170, 170, 170),
              1,
              cv2.LINE_AA,
          )

        cv2.imshow(nome_janela, frame)
        key = cv2.waitKey(1) & 0xFF

        if not self.modo_input:
          if key == ord("q") or key == ord("Q"):
            break
          elif key == ord("c") or key == ord("C"):
            self.modo_input = "cadastro"
            self.nome_digitado = ""
            # Salva exatamente o frame limpo, sem banners ou textos da interface
            self.frame_capturado = frame_limpo.copy()
          elif key == ord("d") or key == ord("D"):
            self.modo_input = "deletar"
            self.nome_digitado = ""
        else:
          if key == 27:
            self.modo_input = None
          elif key == 13:
            nome_alvo = self.nome_digitado.strip()
            if is_cadastro and nome_alvo:
              caminho_salvar = os.path.join(
                  self.diretorio_fotos, f"{nome_alvo}.jpg"
              )
              cv2.imwrite(caminho_salvar, self.frame_capturado)
            elif not is_cadastro and nome_alvo:
              caminho = os.path.join(self.diretorio_fotos, f"{nome_alvo}.jpg")
              if os.path.exists(caminho):
                os.remove(caminho)

            self.treinar_sistema()
            self.modo_input = None
          elif key == 8 or key == 127:
            self.nome_digitado = self.nome_digitado[:-1]
          elif 32 <= key <= 126:
            self.nome_digitado += chr(key)
    finally:
      self.running = False
      if self.conexao_catraca and self.conexao_catraca.is_open:
        self.conexao_catraca.close()
      video_capture.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
  app = SistemaAcessoUlife()
  app.treinar_sistema()
  app.iniciar_reconhecimento()
