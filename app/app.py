import streamlit as st
import pandas as pd
from groq import Groq
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

st.title(" Assistant Logistique")

@st.cache_resource
def load_data():
    # load data
    loader = CSVLoader(file_path=r'C:\Users\voadi\Documents\Voady\test-Data-engineer-voady\data\predictions.csv')
    documents = loader.load()
    # embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
# load vector stor
vectorstore = load_data()

with st.expander("Voir les prédictions"):
    st.dataframe(pd.read_csv(r'C:\Users\voadi\Documents\Voady\test-Data-engineer-voady\data\predictions.csv'))

question = st.text_input("Posez votre question :")

if question:
    with st.spinner("Recherche en cours..."):
        
        # Recherche sémantique dans les données
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Réponse avec Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant logistique expert. Réponds toujours en français."
                },
                {
                    "role": "user",
                    "content": f"""Voici les données pertinentes :
                    {context}

                    Question : {question}"""
                }
            ]
        )
        
        st.write("### Réponse :")
        st.write(response.choices[0].message.content)
        
        with st.expander("Données utilisées pour répondre"):
            st.write(context)