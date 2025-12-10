from deepface import DeepFace
import cv2
import urllib.request
import matplotlib.pyplot as plt

def analisar_imagem(url_imagem):
    try:
        nome_arquivo = "imgPessoa.jpg"
        urllib.request.urlretrieve(url_imagem, nome_arquivo)

        analise = DeepFace.analyze(img_path = nome_arquivo, actions = ['emotion'], enforce_detection=False)

        dados_rosto = analise[0]
        emocao = dados_rosto['dominant_emotion']
        score = dados_rosto['emotion'][emocao] / 100

        print(f"Resultado - Emoção: {emocao.upper()}, Score (confiança): {score:.2f}")
        emocoes = dados_rosto['emotion']

        print("\nProbabilidade por Emoção:")
        for nome, valor in emocoes.items():
          print(f"{nome.capitalize()}: {valor:.2f}%")

        img = cv2.imread(nome_arquivo)
        plt.figure(figsize=(3,3))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Erro ao analisar a imagem: {e}")

print("=== ANÁLISE DA PRIMEIRA IMAGEM ===")
link1 = "https://img.freepik.com/fotos-gratis/homem-vestindo-camiseta-gesticulando_23-2149393651.jpg?semt=ais_hybrid&w=740&q=80"
analisar_imagem(link1)

print("\n=== ANÁLISE DA SEGUNDA IMAGEM ===")
link2 = "https://img.freepik.com/fotos-premium/chorando-homem-triste_102671-4952.jpg?semt=ais_se_enriched&w=740&q=80"
analisar_imagem(link2)

print("\n=== ANÁLISE DA TERCEIRA IMAGEM ===")
link3 = "https://media.istockphoto.com/id/1455764286/pt/foto/celebration-black-woman-and-excited-person-showing-happiness-and-winner-feeling-winning.jpg?s=612x612&w=0&k=20&c=890o0fTxFhvye4qWIR9VyIFLTqyc0mpzVk8vliThOHw="
analisar_imagem(link3)