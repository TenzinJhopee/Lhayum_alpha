import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm

def initialize_client(DB_DIR):
    """Initialize ChromaDB client with version compatibility"""
    try:
        # Try new API first (ChromaDB 0.4+)
        client = chromadb.PersistentClient(path=str(DB_DIR))
        return client, "new_api"
    except Exception as e:
        print(f"New API failed: {e}")
        try:
            # Try old API (ChromaDB 0.3.x)
            settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(DB_DIR)
            )
            client = chromadb.Client(settings)
            return client, "old_api"
        except Exception as e:
            print(f"Old API failed: {e}")
            # Fallback to in-memory
            client = chromadb.Client()
            return client, "memory"


def setup_collection(client, model):
    """Setup or load collection"""
    try:
        # Try to get existing collection
        collection = client.get_collection(
            name="tibetan_qa",
            embedding_function=model
        )
        print(f"✅ Loaded existing collection with {collection.count()} documents")
        return collection
    except:
        # Create new collection
        collection = client.get_or_create_collection(
            name="tibetan_qa",
            embedding_function=model
        )
        print("✅ Created new collection")
        return collection

class TibetanRAG:
    def __init__(self, collection, model):
        self.collection = collection
        self.model = model
        print(f"🚀 TibetanRAG initialized with {collection.count()} documents")
    
    def add_documents(self, excel_file_path):
        """Add documents from Excel file to collection"""
        try:
            df = pd.read_excel(excel_file_path)
            print(f"📊 Loaded Excel with {len(df)} rows")
            df.columns = ['question', 'answer', 'text', 'Unnamed: 3', 'title']
            df = df.dropna(subset=['question', 'text'])
            
            # Create qa_pair column like in your original setup
            df['qa_pair'] = df['question'] + ' ' + df['answer']
            
            # Prepare data lists
            questions = df['question'].tolist()
            answers = df['text'].tolist()
            qa_pairs = df['qa_pair'].tolist()
            
            # Add to collection in batches with explicit embeddings
            batch_size = 100
            total_added = 0
            
            for i in tqdm(range(0, len(questions), batch_size)):
                batch_questions = questions[i:i+batch_size]
                batch_answers = answers[i:i+batch_size]
                batch_qa = qa_pairs[i:i+batch_size]
                
                # Generate embeddings for questions (assuming self.model exists)
                embeddings = self.model.encode(batch_questions)
                
                # Prepare metadatas
                metadatas = [{"question": q, "answer": a} for q, a in zip(batch_questions, batch_answers)]
                
                # Generate IDs
                ids = [f"qa_{j}" for j in range(i, i+len(batch_questions))]
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_qa,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(batch_questions)
                print(f"Added batch {i//batch_size + 1} ({len(batch_questions)} documents)")
            
            print(f"✅ Successfully added {total_added} documents")
            
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
    
    def generate_answer(self, query, n_results=3):
        """Generate answer using RAG"""
        try:
            # Generate query embedding using the same model used for indexing
            query_embedding = self.model.encode([query])
            
            # Query collection with explicit embedding
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
        
            if not results['documents'] or not results['documents'][0]:
                return "སྐྱེས་རོགས། ངས་ཁྱེད་ཀྱི་དྲི་བ་འདི་ལ་ལན་འདེབས་ཐུབ་ཀྱི་མེད། (Sorry, I cannot answer this question.)", results
        
            # Get best matching document
            best_doc = results['documents'][0][0] if results['documents'][0] else ""
            best_metadata = results['metadatas'][0][0] if results['metadatas'][0] else {}
        
            # Create response
            if 'answer' in best_metadata:
                response = best_metadata['answer']
            else:
                response = f"Based on TibetanQA: {best_doc[:200]}..."
        
            return response, results
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)} AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            return error_msg, {"documents": [[]], "metadatas": [[]]}
    
    def get_collection_stats(self):
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {"document_count": count, "status": "ready"}
        except Exception as e:
            return {"document_count": 0, "status": f"error: {e}"}

# # Initialize RAG system
# rag_system = TibetanRAG(collection)

# # Check if we need to load data
# if collection.count() == 0:
#     print("⚠️ Collection is empty. Use rag_system.add_documents('path_to_excel.xlsx') to load data")
# else:
#     print(f"✅ RAG system ready with {collection.count()} documents")