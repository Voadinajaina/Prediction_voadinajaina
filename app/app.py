import streamlit as st
import pandas as pd
from groq import Groq
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Charger le CSV avec LangChain
loader = CSVLoader(file_path='predictions.csv')
documents = loader.load()

# Convertir en contexte texte
context = "\n".join([doc.page_content for doc in documents])

# Interface Streamlit
st.title("🤖 Assistant Logistique")
st.subheader("Analyse de la demande - ThinkPad & AAA Batteries")

with st.expander("Voir les prédictions"):
    st.dataframe(pd.read_csv('predictions.csv'))

question = st.text_input("Posez votre question :")

if question:
    with st.spinner("Analyse en cours..."):
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant logistique expert. Réponds toujours en français de manière claire et concise."
                },
                {
                    "role": "user",
                    "content": f"""Voici les prédictions de demande :
{context}

Question : {question}"""
                }
            ]
        )
        
        st.write("### Réponse :")
        st.write(response.choices[0].message.content)
```

---

### `requirements.txt`
```
streamlit
groq
pandas
langchain
langchain-community
python-dotenv