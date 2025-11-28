from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def score_answer_relevance(question: str, answer: str) -> float:
    """
    Computes relevance score between question and answer using TF-IDF and cosine similarity.
    Returns a float between 0 and 1.
    """
    if not answer or not question:
        return 0.0

    try:
        vectorizer = TfidfVectorizer().fit([question, answer])
        vectors = vectorizer.transform([question, answer])
        qa_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except:
        qa_similarity = 0.0

    return max(0.0, min(qa_similarity, 1.0))
