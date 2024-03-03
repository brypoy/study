########## Search Engine ######################################################################################################
class SearchEngine:
    def __init__(self, documents):
        self.documents = documents

    def search(self, query):
        results = []
        for doc_id, doc_content in enumerate(self.documents):
            if query in doc_content:
                results.append(doc_id)
        return results

# Example usage
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
search_engine = SearchEngine(documents)
query = "document"
search_results = search_engine.search(query)
print("Search results:", search_results)


########## Keyword Based Search ######################################################################################################
class KeywordBasedSearch:
    def __init__(self, documents):
        self.documents = documents

    def search(self, query):
        results = []
        for doc_id, doc_content in enumerate(self.documents):
            if query.lower() in doc_content.lower():
                results.append(doc_id)
        return results

# Example usage
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
keyword_search = KeywordBasedSearch(documents)
query = "Document"
search_results = keyword_search.search(query)
print("Keyword-based search results:", search_results)


########## Vector Retrieval Algorityms ######################################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorBasedSearch:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.vectorized_docs = self.vectorizer.fit_transform(self.documents)

    def search(self, query):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectorized_docs)
        return similarities.argsort()[0][::-1]

# Example usage
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
vector_search = VectorBasedSearch(documents)
query = "second document"
search_results = vector_search.search(query)
print("Vector-based search results:", search_results)


########## Word Embeddings ######################################################################################################
import gensim.downloader as api

class WordEmbeddings:
    def __init__(self, model_name):
        self.model = api.load(model_name)

    def get_embedding(self, word):
        try:
            return self.model[word]
        except KeyError:
            return None

# Example usage
word_embeddings = WordEmbeddings("glove-wiki-gigaword-100")
word = "king"
embedding = word_embeddings.get_embedding(word)
print(f"Word embedding for '{word}': {embedding}")

########## Search Results ######################################################################################################
class SearchResults:
    def __init__(self, documents):
        self.documents = documents

    def display(self, results):
        for doc_id in results:
            print(self.documents[doc_id])

# Example usage
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
search_results = SearchResults(documents)
search_results.display([0, 2])
