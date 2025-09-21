import streamlit as st
from rag_systems import TibetanRAG, setup_collection, initialize_client
from pathlib import Path
from sentence_transformers import SentenceTransformer
# import chromadb
# # from chromadb.config import Settings
# from tqdm import tqdm
# from lib_client import llm


# # Initialize ChromaDB
# client = chromadb.PersistentClient(path="./tibetan_qa_db")
# collection = client.create_collection("tibetan_qa")
# llm = ApertusSwissLLM(api_key="sk-768c82ef24604a4db381bf8588a73007")

def main():
    st.set_page_config(
        page_title="Tibetan RAG Chatbot",
        page_icon="🏔️",
        layout="wide"
    )

    # Database configuration
    DB_DIR = Path("./tibetan_qa_db")
    DB_DIR.mkdir(exist_ok=True)

    # Initialize client
    client, api_type = initialize_client(DB_DIR)
    print(f"Using ChromaDB with {api_type}")

    # Load embedding model
    model = SentenceTransformer('sentence-transformers/LaBSE')
    print("✅ Loaded LaBSE model")

    collection = setup_collection(client, model)
    rag_system = TibetanRAG(collection, model)
    # rag_system.add_documents('TibetanQA.xlsx')
    
    st.title("🏔️ Tibetan RAG Chatbot")
    st.markdown("*AI-powered Tibetan Question Answering System*")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("དྲི་བ་གནང་རོགས། (Please ask your question)"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("སྒྲུབ་བཞིན་པ། (Processing...)"):
                response = rag_system.generate_answer(prompt)
                st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.info(
            "This chatbot uses RAG (Retrieval-Augmented Generation) "
            "to answer questions in Tibetan based on the TibetanQA dataset."
        )
        
        st.header("Dataset Info")
        st.metric("Articles", "~1,513")
        st.metric("Q&A Pairs", "~20,000")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()