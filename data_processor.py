import openai
import pandas as pd
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def enrich_course_data_with_openai(df):
    """
    Usa a OpenAI para enriquecer os dados dos cursos, gerando descrições ou categorias.
    """
    for index, row in df.iterrows():
        try:
            prompt = f"Crie uma breve descrição para o curso: {row['title']}."
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.7,
            )
            # Adicionar a descrição gerada ao DataFrame
            df.at[index, 'description'] = response.choices[0].text.strip()
        except Exception as e:
            print(f"Erro ao gerar descrição para o curso '{row['title']}': {e}")
            df.at[index, 'description'] = "Descrição não disponível."
    return df

def load_and_preprocess_data():
    # Verificar se o arquivo existe
    file_path = 'data/cursos.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")

    # Carregar os dados
    df = pd.read_csv(file_path)

    # Exibir as primeiras linhas do DataFrame para depuração
    print("Colunas disponíveis no CSV:", df.columns)
    print("Primeiras linhas do DataFrame:")
    print(df.head())

    # Verificar se as colunas necessárias estão presentes
    required_columns = ['id', 'title', 'autor', 'notas', 'participantes']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"A coluna {col} não está presente no arquivo CSV.")

    # Garantir que 'notas' e 'participantes' sejam numéricos
    df['notas'] = pd.to_numeric(df['notas'], errors='coerce').fillna(0)
    df['participantes'] = pd.to_numeric(df['participantes'], errors='coerce').fillna(0)

    # Preprocessar os dados
    df.fillna(0, inplace=True)  # Preencher valores ausentes com 0

    # Criar uma coluna 'feature' combinando informações relevantes
    df['feature'] = df['notas'] * df['participantes']

    # Exibir o DataFrame após o preprocessamento
    print("DataFrame após o preprocessamento:")
    print(df.head())

    df = enrich_course_data_with_openai(df)

    return df