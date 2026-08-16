import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    logging.error("scikit-learn is required. Please install it with 'pip install scikit-learn'.")
    raise

logger = logging.getLogger(__name__)

class DocumentCatalog:
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.resumes_df = None
        
        # We define latent topics to ensure co-occurrence for LSA
        self.topics = {
            "data_engineering": ["etl", "data pipelines", "airflow", "spark", "data warehouse", "big query"],
            "frontend_dev": ["react", "javascript", "ui", "frontend", "css", "html", "web design"],
            "backend_dev": ["java", "spring", "backend", "apis", "microservices", "sql", "database"],
            "ml_ai": ["machine learning", "deep learning", "neural networks", "nlp", "computer vision", "pytorch"]
        }
        
    def generate_catalog(self, num_docs=1000):
        docs = []
        for i in range(num_docs):
            topic = self.rng.choice(list(self.topics.keys()))
            words = self.topics[topic]
            
            # Select 3-5 random phrases from the topic
            doc_phrases = self.rng.choice(words, size=self.rng.integers(3, 6), replace=False)
            
            # Add some random noise phrases from other topics to connect them slightly
            if self.rng.random() < 0.2:
                other_topic = self.rng.choice(list(self.topics.keys()))
                noise = self.rng.choice(self.topics[other_topic], size=1)
                doc_phrases = np.append(doc_phrases, noise)
                
            text = " ".join(doc_phrases)
            docs.append({
                "doc_id": f"res_{i:04d}",
                "topic": topic,
                "text": text
            })
            
        self.resumes_df = pd.DataFrame(docs)
        self.resumes_df.set_index("doc_id", inplace=True)
        logger.info(f"Generated {num_docs} synthetic resumes.")

class SemanticSearchEngine:
    def __init__(self, catalog: DocumentCatalog, n_components: int = 4):
        self.catalog = catalog
        self.n_components = min(n_components, len(self.catalog.resumes_df))
        
        # TF-IDF (Keyword)
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = None
        
        # LSA (Semantic)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.semantic_matrix = None
        
    def build_index(self):
        logger.info("Building TF-IDF and Semantic SVD index...")
        texts = self.catalog.resumes_df['text'].tolist()
        
        # Keyword Index
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Semantic Index (LSA)
        self.semantic_matrix = self.svd.fit_transform(self.tfidf_matrix)
        
        # Normalize semantic vectors for fast cosine similarity
        norms = np.linalg.norm(self.semantic_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.semantic_matrix = self.semantic_matrix / norms
        logger.info("Index built.")

    def search(self, query: str, method: str = "hybrid", alpha: float = 0.8, k: int = 10) -> List[Dict]:
        """
        method: 'keyword', 'semantic', 'hybrid'
        alpha: weight of semantic score (0.0 to 1.0)
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Index not built.")
            
        query_tfidf = self.tfidf.transform([query])
        
        keyword_scores = np.zeros(len(self.catalog.resumes_df))
        semantic_scores = np.zeros(len(self.catalog.resumes_df))
        
        # 1. Keyword Score
        if method in ["keyword", "hybrid"]:
            keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
        # 2. Semantic Score
        if method in ["semantic", "hybrid"]:
            query_semantic = self.svd.transform(query_tfidf)
            q_norm = np.linalg.norm(query_semantic, axis=1, keepdims=True)
            q_norm[q_norm == 0] = 1.0
            query_semantic = query_semantic / q_norm
            
            # dot product of normalized vectors = cosine similarity
            semantic_scores = self.semantic_matrix.dot(query_semantic.T).flatten()
            # map to [0, 1] to keep it somewhat positive and comparable to TF-IDF
            semantic_scores = (semantic_scores + 1.0) / 2.0
            
        # 3. Combine
        if method == "keyword":
            final_scores = keyword_scores
        elif method == "semantic":
            final_scores = semantic_scores
        else: # hybrid
            final_scores = (alpha * semantic_scores) + ((1.0 - alpha) * keyword_scores)
            
        # Top K
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        results = []
        doc_ids = self.catalog.resumes_df.index
        texts = self.catalog.resumes_df['text'].values
        
        for idx in top_indices:
            score = float(final_scores[idx])
            # Drop zero keyword matches if using pure keyword
            if method == "keyword" and score == 0:
                continue
                
            doc_id = doc_ids[idx]
            text = texts[idx]
            
            # Explainability
            if method == "keyword":
                explanation = f"Exact keyword match. (KW Score: {score:.2f})"
            elif method == "semantic":
                explanation = f"Conceptually similar. (Sem Score: {score:.2f})"
            else:
                explanation = f"Hybrid match (KW: {keyword_scores[idx]:.2f}, Sem: {semantic_scores[idx]:.2f})"
                
            results.append({
                "doc_id": doc_id,
                "text": text,
                "score": score,
                "explanation": explanation
            })
            
        return results
