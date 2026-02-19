# Image de base Python
FROM python:3.12-slim

# Répertoire de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le dossier app et data
COPY app/ ./app/
COPY data/ ./data/

# Copier le fichier .env
COPY app/.env .env

# Exposer le port Streamlit
EXPOSE 8501

# Lancer l'application depuis le dossier app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]