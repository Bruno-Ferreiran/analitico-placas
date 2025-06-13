# OCR de Placas com Python (Condomínio Marina Ilha Verde)
Este projeto realiza a leitura de placas de veículos em tempo real através de uma câmera RTSP. A detecção é feita com o modelo YOLO (Ultralytics), e o reconhecimento dos caracteres da placa (OCR) é feito com a biblioteca EasyOCR.

---

# Contexto
Consegui acesso à câmera do condomínio Marina Ilha Verde, e configurei o sistema para capturar o fluxo de vídeo em tempo real.

Se quiser visualizar a câmera ao vivo, é possível através do seguinte link:
https://vision.chip7.cc/#/cembed/0365187776062249225e60c53d4146b21fb741bd6b1a04ac81177aefce50746ee5a724257221e526bb32f9826630

---

# Pré-requisitos:
- Python 3.10 ou superior
- Uma máquina com GPU NVIDIA (opcional – roda também em CPU)
- FFmpeg instalado no sistema e adicionado ao PATH

# Instalar dependências:
- pip install -r requirements.txt

# Executar o script:
- python captura_rtsp.py 

# Feito por Bruno Ferreira e Edison Hernandez

# Funcionalidades:
- Leitura do vídeo ao vivo via RTSP usando FFmpeg
- Detecção de placas com modelo YOLOv8 customizado
- Reconhecimento de caracteres com EasyOCR (português e inglês)
- Limpeza automática de texto e formatação no padrão AAA-0000
- Ignora placas repetidas consecutivas para evitar duplicações
- Gera prints automáticos a cada 10 segundos para debug
- Salva imagens de placas reconhecidas com nome e horário
- Salva placas **não reconhecidas** em uma pasta separada
- Exibe log do OCR com textos brutos lidos e candidatos de placa
- Exibe tempo de loop de processamento a cada frame
- Compatível com CPU ou GPU (auto detecta)
- Pode ser encerrado com Ctrl + C

# Detalhe
a cada 10 segundos, o sistema foi configurado para tirar automaticamente uma captura da tela (print) como forma de debug. Isso permite verificar se o vídeo está travado ou com delay. Essas imagens são salvas na pasta: output/snapshots

- Já as imagens das placas detectadas com sucesso pelo OCR são salvas em: output/plates

- As placas que **não foram reconhecidas corretamente** pelo OCR são salvas em: output/plates_erros

