**parser.py**

def parse_resume_text(text):
    """
    Simulates resume parsing. Can be replaced with actual PDF/DOCX parser.
    """
    return text.strip()

"""**Matcher.py**"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ResumeMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.job_descriptions = []
        self.job_titles = []

    def fit(self, job_descriptions, job_titles):
        """
        Train the vectorizer on job descriptions.
        """
        self.job_descriptions = job_descriptions
        self.job_titles = job_titles
        self.job_vectors = self.vectorizer.fit_transform(job_descriptions)

    def predict(self, resumes):
        """
        Match each resume to the most relevant job description.
        """
        resume_vectors = self.vectorizer.transform(resumes)
        results = []
        for i, rv in enumerate(resume_vectors):
            scores = cosine_similarity(rv, self.job_vectors)[0]
            best_idx = scores.argmax()
            results.append({
                'Resume': resumes[i],
                'Best Match': self.job_titles[best_idx],
                'Similarity Score': round(scores[best_idx], 2)
            })
        return pd.DataFrame(results)

"""**Preprocess.py**"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """
    Preprocess the input text: lowercase, remove punctuation/numbers, tokenize, lemmatize.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)
