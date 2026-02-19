import streamlit as st
import pandas as pd
from groq import Groq
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = r'C:\Users\voadi\Documents\Voady\test-Data-engineer-voady\data\predictions.csv'
FAISS_INDEX_PATH = r'C:\Users\voadi\Documents\Voady\test-Data-engineer-voady\data\faiss_index'

st.title("🤖 Assistant Logistique")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        loader = CSVLoader(file_path=DATA_PATH)
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

@st.cache_data
def load_summary():
    df = pd.read_csv(DATA_PATH)
    summary = df.groupby(['location', 'Product'])['predicted_demand'].sum().reset_index()
    summary = summary.sort_values('predicted_demand', ascending=False)
    return df, summary

vectorstore = load_vectorstore()
df_pred, summary = load_summary()

# with st.expander("Voir les prédictions"):
#     st.dataframe(df_pred)

question = st.text_input("Posez votre question :")

if question:
    with st.spinner("Recherche en cours..."):

        # Contexte 1 : résumé global toutes les villes
        global_context = summary.to_string(index=False)

        # Contexte 2 : recherche sémantique FAISS
        docs = vectorstore.similarity_search(question, k=10)
        semantic_context = "\n".join([doc.page_content for doc in docs])

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Tu es un assistant logistique expert.
Réponds toujours en français de manière claire et concise.
Utilise TOUTES les villes disponibles pour comparer et répondre.
Ne te limite pas à une seule ville."""
                },
                {
                    "role": "user",
                    "content": f"""Voici le résumé global de toutes les villes :
{global_context}

Voici les données détaillées pertinentes :
{semantic_context}

Question : {question}"""
                }
            ],
            max_tokens=800
        )

        st.write("### Réponse :")
        st.write(response.choices[0].message.content)

        with st.expander("Données utilisées pour répondre"):
            st.write(semantic_context)