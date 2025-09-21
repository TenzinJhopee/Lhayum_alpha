from pathlib import Path
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm
from lib_client import llm
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
DB_DIR = (BASE_DIR / 'tibetan_qa_db').resolve()


# Initialize ChromaDB
client = Client(Settings(chroma_db_impl = 'duckdb+parquet', persistent_directory=str(DB_DIR)))
model = SentenceTransformer('sentence-transformers/LaBSE')
collection = client.get_collection(
    name="tibetan_qa_db",
    embedding_function=model
)
# llm = ApertusSwissLLM(api_key="sk-768c82ef24604a4db381bf8588a73007")


class TibetanRAGSystem:
    def __init__(self, collection, embedding_model, llm):
        self.collection = collection
        self.embedding_model = embedding_model
        self.llm = llm
        
    def retrieve_relevant_qa(self, query, n_results=5):
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return results
    
    def generate_answer(self, query):
        # Retrieve relevant Q&A pairs
        relevant_qa = self.retrieve_relevant_qa(query)
        
        # Prepare context from retrieved documents
        context = ""
        for i, doc in enumerate(relevant_qa['documents'][0]):
            metadata = relevant_qa['metadatas'][0][i]
            context += f"Q: {metadata['question']}\nA: {metadata['answer']}\n\n"
        
        # Generate system prompt for Tibetan
        system_prompt = f"""
        You are a helpful assistant that answers questions in Tibetan based on the provided context.
        Use the following question-answer pairs as reference to answer the user's question.
        If you cannot find relevant information in the context, politely say so in Tibetan.
        
        Context:
        {context}
        """
        
        # Generate response
        response = self.llm.generate_response(query, context=system_prompt)
        return response, relevant_qa

# Initialize RAG system
rag_system = TibetanRAGSystem(collection, model, llm)