import os
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class SimplePDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        docs = []
        with open(self.file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            docs.append(Document(page_content=text, metadata={"source": self.file_path}))
        return docs

class SimpleDirectoryLoader:
    def __init__(self, dir_path, glob="**/*.pdf"):
        self.dir_path = dir_path
        self.glob = glob
        
    def load(self):
        documents = []
        for root, dirs, files in os.walk(self.dir_path):
            for file in files:
                if file.endswith(".pdf"):
                    loader = SimplePDFLoader(os.path.join(root, file))
                    documents.extend(loader.load())
                elif file.endswith(".txt"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        text = f.read()
                        documents.append(Document(page_content=text, metadata={"source": os.path.join(root, file)}))
        return documents

import numpy as np

class SimpleVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        texts = [doc.page_content for doc in documents]
        new_vectors = self.embeddings.embed_documents(texts)
        if not self.vectors:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

    @classmethod
    def from_documents(cls, documents, embeddings):
        instance = cls(embeddings)
        instance.add_documents(documents)
        return instance

    def save_local(self, folder_path):
        import pickle
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(os.path.join(folder_path, "index.pkl"), "wb") as f:
            pickle.dump({"docs": self.documents, "vecs": self.vectors}, f)

    @classmethod
    def load_local(cls, folder_path, embeddings):
        import pickle
        with open(os.path.join(folder_path, "index.pkl"), "rb") as f:
            data = pickle.load(f)
        instance = cls(embeddings)
        instance.documents = data["docs"]
        instance.vectors = data["vecs"]
        return instance

    def similarity_search(self, query, k=3):
        query_vec = self.embeddings.embed_query(query)
        # Cosine similarity
        scores = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        )
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

    def as_retriever(self, search_kwargs={"k": 3}):
        class SimpleRetriever:
            def __init__(self, store, k):
                self.store = store
                self.k = k
            def invoke(self, query):
                return self.store.similarity_search(query, k=self.k)
        return SimpleRetriever(self, search_kwargs.get("k", 3))

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self, data_dir="data", index_path="faiss_index"):
        self.data_dir = data_dir
        self.index_path = index_path
        # Using OpenRouter for embeddings to avoid local Torch/DLL issues
        self.embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            openai_api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
            openai_api_base=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
        )
        self.vector_store = None
        
    def ingest_documents(self):
        """Load and index documents from the data directory."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        print(f"Loading documents from {self.data_dir}...")
        
        loader = SimpleDirectoryLoader(self.data_dir)
        docs = loader.load()
            
        if not docs:
            print("No documents found to index.")
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(docs)
        
        print(f"Creating vector store with {len(splits)} splits...")
        self.vector_store = SimpleVectorStore.from_documents(splits, self.embeddings)
        self.vector_store.save_local(self.index_path)
        return self.vector_store

    def load_index(self):
        """Load the vector store from local storage."""
        if os.path.exists(os.path.join(self.index_path, "index.pkl")):
            self.vector_store = SimpleVectorStore.load_local(
                self.index_path, 
                self.embeddings
            )
            return self.vector_store
        return None

    def get_retriever(self, k=3):
        if not self.vector_store:
            self.load_index()
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        return None
