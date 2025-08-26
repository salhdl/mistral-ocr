import os
import time
import faiss
import numpy as np
from mistralai import Mistral
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("❌ MISTRAL_API_KEY n'est pas défini dans l'environnement ou le fichier .env")

client = Mistral(api_key=api_key)

def ocr_pdf():
    uploaded_pdf = client.files.upload(
        file={
            "file_name": "light-duty-vehicules.pdf",
            "content": open("pdf/light-duty-vehicules.pdf", "rb"),
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    print("Starting OCR request...")
    # 🔹 On enlève include_image_base64 pour éviter d’avoir du binaire dans le fichier
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )
    print("OCR request completed")

    with open("ocr_response.md", "w", encoding="utf-8") as f:
        f.write("\n".join([page.markdown for page in ocr_response.pages]))

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings.create(
          model="mistral-embed",
          inputs=input
      )
    return embeddings_batch_response.data[0].embedding

def run_mistral(user_message, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)

def main():
    ocr_pdf()

    # 🔹 Lecture sécurisée du fichier OCR avec remplacement des caractères invalides
    with open("ocr_response.md", "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Chunk du texte en morceaux de 2048 caractères
    chunk_size = 2048
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Embeddings pour chaque chunk
    text_embeddings = []
    for chunk in chunks:
        embedding = get_text_embedding(chunk)
        text_embeddings.append(embedding)
        time.sleep(2)  # éviter rate limit

    text_embeddings = np.array(text_embeddings)

    # Index FAISS
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Question utilisateur
    question = input('Enter your question: ')
    question_embeddings = np.array([get_text_embedding(question)])

    # Recherche des chunks les plus proches
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

    # Prompt pour Mistral; What are the CO2 emission targets for light-duty vehicles?

    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """

    response = run_mistral(prompt)
    print('Response: ', response)

if __name__ == "__main__":
    main()
