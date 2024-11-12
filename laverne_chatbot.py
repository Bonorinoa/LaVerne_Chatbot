import streamlit as st
import time
import datetime as dt
import json
import numpy as np
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

from utils import RAGConfig, RAG, RetrieverConfig, RetrieverType
from langchain_core.prompts import PromptTemplate

PERSIST_DIRECTORY = "./chroma_langchain_db"  # Updated to use relative path compatible with Streamlit Cloud
COLLECTION_NAME = "ulv_handbook"
KNOWLEDGE_BASE_PATH = "ulv-faculty-handbook-23-24.pdf"  # Corrected file name format

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
# Store current query (i.e., last user input)
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
    
# Store the time of first interaction (when they opened the app) 
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = time.time()
    
# Store cumulative time spent over the session.
if "time_spent" not in st.session_state:
    st.session_state.time_spent = 0
    
# callback to record time until feedback
def record_time_until_feedback():
    st.session_state.time_until_feedback = time.time() - st.session_state.session_start_time
    
# This function will run after each interaction (i.e., clicks any button) to keep track of total seconds spent in the session.
# I don't think is the best way to do it, but it works as a proxy for now. 
def acumulate_time():
    st.session_state.time_spent += time.time() - st.session_state.session_start_time
    st.session_state.session_start_time = time.time()

    
# Store feedback
if "feedback" not in st.session_state:
    st.session_state.feedback = ""        
    
# callback to store feedback in text file
def record_feedback():
    with open("feedback.txt", "a") as f:
        f.write(f"FEEDBACK: \n\n {st.session_state.feedback} \n\n ------------------ \n\n SUBMISSION DATE: {dt.datetime.today()} \n")
        f.write("\n")

# callback to store feedback, submission date, chat history, and total cost in a json file
def record_feedback_json():
    dict_feedback = {"Random_id": np.random.randint(10000),
                     "Content": {
                                "feedback": st.session_state.feedback,
                                "chat_history": st.session_state.messages,
                                "total_cost": st.session_state.total_cost,
                                "time_until_feedback": st.session_state.time_spent,
                                "submission_date": dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                                }
                     }
    
    json_feedback = json.dumps(dict_feedback, indent=2)
    
    with open("feedback.json", "a") as f:
        f.write(json_feedback)
        f.write("\n")
        
    st.sidebar.write("Feedback stored in feedback.json")
    

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", 
                                  "content": "How may I assist you today?"}]

def get_chroma_settings():
    """Get ChromaDB settings with persistence"""
    return Settings(
        persist_directory=PERSIST_DIRECTORY,
        is_persistent=True,
        anonymized_telemetry=False
    )
    
# Initialize RAG system
def initialize_rag() -> RAG:
    """Initialize the RAG system with configuration and persistent storage"""
    if "rag_system" not in st.session_state:
        try:
            # Configure the RAG system
            config = RAGConfig(
                chunk_size=1000,
                chunk_overlap=200,
                temperature=0.0,
                max_tokens=350,
                retrievers=[
                    RetrieverConfig(
                        retriever_type=RetrieverType.SIMILARITY,
                        top_k=3,
                        weight=0.6
                    ),
                    RetrieverConfig(
                        retriever_type=RetrieverType.BM25,
                        top_k=3,
                        weight=0.4
                    )
                ]
            )
            
            # Ensure persistence directory exists
            Path(PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
            
            # Create RAG instance with persistence directory and collection name
            rag = RAG(config, persist_directory=PERSIST_DIRECTORY, collection_name=COLLECTION_NAME)
            
            # Define the chat prompt template
            prompt_template = PromptTemplate(
                template="""Based on the following context from the University of La Verne Faculty Handbook, 
                provide a helpful and natural response to the user's question. 
                If the context doesn't contain relevant information, say so politely.

                Context: {context}

                User Question: {user_query}

                Response:""",
                input_variables=["context", "user_query"]
            )
            
            # Check if collection exists
            if not check_vectorstore_exists(PERSIST_DIRECTORY, COLLECTION_NAME):
                st.info("Initializing knowledge base for the first time. This may take a few minutes...")
                rag.initialize_full_pipeline(KNOWLEDGE_BASE_PATH, prompt_template)
            else:
                st.info("Loading existing knowledge base...")
                rag.load_existing_vectorstore(PERSIST_DIRECTORY, prompt_template)
            
            st.session_state.rag_system = rag
            st.session_state.rag_initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {str(e)}")
            st.session_state.rag_initialized = False
            raise
            
    return st.session_state.rag_system


def llm_response_generator(llm_response):
    """
    Generator function that yields formatted parts of the LLM response.
    It splits the response into paragraphs using newline characters and yields each paragraph.
    This helps in maintaining the structured output for display in the Streamlit chat interface.
    """
    paragraphs = llm_response.split('\n\n')  # Split on double newline for distinct sections
    for paragraph in paragraphs:
        if paragraph.strip():  # Only yield non-empty paragraphs
            yield paragraph
            time.sleep(0.1)
        yield '\n'

def get_rag_response(query: str, rag: Optional[RAG] = None) -> str:
    """Get response from RAG system with error handling and reconnection logic"""
    try:
        if rag is None:
            raise ValueError("RAG system not initialized")
            
        # Log the query for debugging
        st.session_state.get('query_log', []).append({
            'timestamp': dt.datetime.now().isoformat(),
            'query': query
        })
        
        try:
            response = rag.query(query)
        except Exception as e:
            # If there's an error, try to reinitialize the RAG system
            st.warning("Reconnecting to knowledge base...")
            # Clear the session state to force reinitialization
            if "rag_system" in st.session_state:
                del st.session_state.rag_system
            # Get a fresh RAG instance
            rag = initialize_rag()
            response = rag.query(query)
            
        return response
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.error(error_msg)
        return error_msg

def check_vectorstore_exists(persist_dir: str, collection_name: str) -> bool:
    """Check if a Chroma vectorstore collection exists in the given directory"""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        return any(col.name == collection_name for col in collections)
    except Exception:
        return False

def run_bot():
    st.set_page_config(page_title="La Verne Chabot")
   
    # Initialize RAG system
    rag = initialize_rag()
    
    with st.sidebar:
        st.title('🤗💬 La Verne Chabot')
        
        # Add RAG configuration options in sidebar
        if st.checkbox("Show Advanced Settings"):
            st.write("Retriever Settings")
            similarity_weight = st.slider("Similarity Retriever Weight", 0.0, 1.0, 0.6, 0.1)
            bm25_weight = 1.0 - similarity_weight
            st.write(f"BM25 Retriever Weight: {bm25_weight:.1f}")
            
            top_k = st.slider("Number of Retrieved Documents", 1, 10, 3)
            
            # Update RAG configuration if changed
            if st.button("Update Configuration"):
                new_config = RAGConfig(
                    retrievers=[
                        RetrieverConfig(
                            retriever_type=RetrieverType.SIMILARITY,
                            top_k=top_k,
                            weight=similarity_weight
                        ),
                        RetrieverConfig(
                            retriever_type=RetrieverType.BM25,
                            top_k=top_k,
                            weight=bm25_weight
                        )
                    ]
                )
                rag.update_config(**new_config.__dict__)
                st.success("Configuration updated!")
        
        st.sidebar.subheader("Feedback")
        st.sidebar.write("Was the conversation helpful? Your honest feedback will help me improve the system.")
        feedback = st.sidebar.text_area("Feedback", height=75)
        if st.sidebar.button("Submit Feedback", on_click=record_time_until_feedback):
            st.session_state.feedback = feedback
            st.sidebar.success("Thank you for your feedback!")
            record_feedback_json()
            
        st.sidebar.write(f"Cumulative time spent: {round(st.session_state.time_spent, 2)} seconds")
            
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input(on_submit=acumulate_time):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.current_query = prompt

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response from RAG system
                response = get_rag_response(st.session_state.current_query, rag)
                st.write_stream(llm_response_generator(response))
                
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    response_start_time = time.time()
    run_bot()
    response_end_time = time.time()
    print(f"Response time: {response_end_time - response_start_time}")