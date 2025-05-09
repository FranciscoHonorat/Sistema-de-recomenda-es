from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def train_model(df):
    # Criar a matriz TF-IDF com base na coluna 'feature'
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['feature'].astype(str))  # Converter para string

    # Calcular a similaridade do cosseno
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim