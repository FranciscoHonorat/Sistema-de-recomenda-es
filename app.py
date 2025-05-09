from flask import Flask, render_template, request
from data_processor import load_and_preprocess_data
from recommender import train_model
import pandas as pd
import openai
import os

#Configuração da chave da API do OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_recommendations(keyword):
    try:
        prompt = f"Recomende cursos relacionados à palavra-chave '{keyword}' com base nos dados disponíveis."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Erro ao chamar a API do OpenAI: {e}")
        return None

app = Flask(__name__)

try:
    # Carrega dados e modelo de uma vez no início
    df = load_and_preprocess_data()
    cosine_sim = train_model(df)
except Exception as e:
    print(f"Erro ao carregar os dados ou treinar o modelo: {e}")
    df = None
    cosine_sim = None

@app.route('/')
def home():
    try:
        courses = df['title'].tolist()
        return render_template('index.html', courses=courses)
    except Exception as e:
        return f"Erro ao carregar os dados: {e}", 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Obter a palavra-chave fornecida pelo usuário
        keyword = request.form.get('keyword', '').strip().lower()

        # Verificar se a palavra-chave foi fornecida
        if not keyword:
            return "Erro: Nenhuma palavra-chave foi fornecida.", 400

        # Filtrar os cursos com base na palavra-chave
        filtered_df = df[df['title'].str.lower().str.contains(keyword, na=False)]

        print("Cursos filtrados:", filtered_df)

        # Verificar se há cursos correspondentes
        if filtered_df.empty:
            # Usar a OpenAI para gerar uma recomendação
            openai_recommendation = get_openai_recommendation(keyword)
            if openai_recommendation:
                return render_template(
                    'index.html',
                    keyword=keyword,
                    recommended_course={
                        'title': openai_recommendation,
                        'autor': "OpenAI",
                        'participantes': "N/A",
                        'notas': "N/A",
                        'feature': "N/A"
                    }
                )
            else:
                return f"Erro: Nenhum curso encontrado para a palavra-chave '{keyword}', e a OpenAI não pôde gerar uma recomendação.", 400

        # Ordenar os cursos filtrados pela feature (descendente)
        recommended_course = filtered_df.sort_values(by='feature', ascending=False).iloc[0]

        # Obter informações do curso recomendado
        recommended_course_info = {
            'title': recommended_course['title'],
            'autor': recommended_course['autor'],
            'participantes': recommended_course['participantes'],
            'notas': recommended_course['notas'],
            'feature': recommended_course['feature']
        }

        # Passar os dados para o template
        return render_template(
            'index.html',
            keyword=keyword,
            recommended_course=recommended_course_info
        )

    except Exception as e:
        return f"Erro ao processar a recomendação: {e}", 500
    
