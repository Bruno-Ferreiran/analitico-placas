# OCR de Placas com Python (Condomínio Marina Ilha Verde)
Este projeto realiza a leitura de placas de veículos em tempo real através de uma câmera RTSP. A detecção é feita com o modelo YOLO (Ultralytics), e o reconhecimento dos caracteres da placa (OCR) é feito com a biblioteca EasyOCR.

---

# Contexto
Consegui acesso à câmera do condomínio Marina Ilha Verde, e configurei o sistema para capturar o fluxo de vídeo em tempo real.

Se  quiser visualizar a câmera ao vivo, é possível através do seguinte link:
 https://vision.chip7.cc/#/cembed/0365187776062249225e60c53d4146b21fb741bd6b1a04ac81177aefce50746ee5a724257221e526bb32f9826630

--- 

# Pré-requisitos:
- Python 3.10 ou superior
- Uma máquina com GPU NVIDIA
- FFmpeg instalado no sistema

# Instalar dependências:
- pip install -r requirements.txt

# Executar o script:
- python captura_rtsp.py 

# Feito por Bruno Ferreira e Edison Hernandez

# Detalhe
a cada 10 segundos, o sistema foi configurado para tirar automaticamente uma captura da tela (print) como forma de debug. Isso permite verificar se o vídeo está travado ou com delay. Essas imagens são salvas na pasta: output/snapshots

- Já as imagens das placas detectadas com sucesso pelo OCR são salvas em: output/plates

