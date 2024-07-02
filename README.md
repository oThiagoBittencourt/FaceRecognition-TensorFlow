# Reconhecimento Facial com TensorFlow e OpenCV 🕵️‍♂️
![faceRecognition](https://github.com/oThiagoBittencourt/FaceRecognition-TensorFlow/assets/106789198/8f0b596c-2f30-4324-afe8-85f7a0838190)

Este projeto implementa um sistema de reconhecimento facial usando TensorFlow, OpenCV e SVM.

## Características
- Detecção de rostos e extração de características faciais.
- Treinamento de um modelo SVM para reconhecimento facial.
- Previsão de rostos em tempo real.

## Requisitos
- Python 3.7+
- OpenCV
- TensorFlow
- scikit-learn
- joblib

---

### Fase 1: Captura de Imagens
Na fase 1, captura imagens da webcam para serem utilizadas no treinamento do modelo. Cada imagem é processada para detectar o rosto e extrair características faciais (olhos e boca). As imagens são então armazenadas em um diretório específico para cada usuário.

### Fase 2: Treinamento do Modelo
Na fase 2, utiliza as imagens capturadas na fase 1 para treinar um modelo de máquina de vetor de suporte (SVM) para reconhecimento facial. Extrai as características das imagens e as utiliza para treinar o modelo SVM, que é então salvo para uso posterior.

### Fase 3: Previsão em Tempo Real
Na fase 3, utiliza o modelo treinado para prever rostos em tempo real usando a webcam. O sistema exibe o nome do usuário reconhecido ou "Desconhecido" se a confiança for inferior a 75%.

---

### Alunos:
- Thiago Bittencourt Santana
- Gabriel Martins Delfes
