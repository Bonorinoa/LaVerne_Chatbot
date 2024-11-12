from langchain_groq import ChatGroq

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.retrievers import EnsembleRetriever

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, ConfigurableField
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.retrievers import BM25Retriever

from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging
import os
from time import perf_counter
from functools import wraps

import chromadb
from chromadb.config import Settings

import streamlit as st

os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def timer_decorator(func):
    """Decorator to measure and log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = perf_counter()
            logging.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = perf_counter()
            logging.error(f"{func.__name__} failed after {end_time - start_time:.2f} seconds")
            raise
    return wrapper

class RetrieverType(Enum):
    """Supported retriever types"""
    SIMILARITY = "similarity"
    BM25 = "bm25"
    ENSEMBLE = "ensemble"

@dataclass
class RetrieverConfig:
    """Configuration for retrievers"""
    retriever_type: RetrieverType
    top_k: int = 4
    weight: float = 1.0  # Used for ensemble retriever

    def __post_init__(self):
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.weight < 0 or self.weight > 1:
            raise ValueError("weight must be between 0 and 1")

@dataclass
class RAGConfig:
    """Configuration for RAG components"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3-8b-8192"
    temperature: float = 0.0
    max_tokens: int = 250
    retrievers: List[RetrieverConfig] = field(default_factory=lambda: [
        RetrieverConfig(RetrieverType.SIMILARITY, top_k=4, weight=1.0)
    ])

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        
        # Validate retriever weights sum to 1.0 if using ensemble
        if any(r.retriever_type == RetrieverType.ENSEMBLE for r in self.retrievers):
            total_weight = sum(r.weight for r in self.retrievers)
            if not (0.99 <= total_weight <= 1.01):  # Allow for floating-point imprecision
                raise ValueError("Retriever weights must sum to 1.0")

class RAG:
    """
    Modular implementation of Retrieval Augmented Generation with logging and error handling
    """
    def __init__(self, config: Optional[RAGConfig] = None, persist_directory: Optional[str] = "./chroma_db", collection_name: str = "ulv_handbook"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing RAG pipeline")
        
        try:
            self.config = config or RAGConfig()
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.documents = []
            self.splits = []
            self.embeddings = None
            self.vectorstores = {}
            self.retrievers = {}
            self.ensemble_retriever = None
            self.llm = None
            self.chain = None
            
            # Initialize persistent Chroma client if persistence is enabled
            if persist_directory:
                self.chroma_client = chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG: {str(e)}")
            raise

    @timer_decorator
    def load_document(self, file_path: Union[str, Path]) -> None:
        """Load document based on file extension"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            loaders = {
                '.pdf': PyPDFLoader,
                '.txt': TextLoader,
                '.csv': CSVLoader
            }

            loader_cls = loaders.get(file_path.suffix.lower())
            if not loader_cls:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            self.logger.info(f"Loading document: {file_path}")
            loader = loader_cls(str(file_path))
            self.documents = loader.load()
            self.logger.info(f"Successfully loaded {len(self.documents)} documents")

        except Exception as e:
            self.logger.error(f"Error loading document: {str(e)}")
            raise

    @timer_decorator
    def split_text(self) -> None:
        """Split documents into chunks"""
        try:
            if not self.documents:
                raise ValueError("No documents loaded")

            self.logger.info("Splitting documents into chunks")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                add_start_index=True
            )
            self.splits = splitter.split_documents(self.documents)
            self.logger.info(f"Created {len(self.splits)} chunks")

        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            raise

    @timer_decorator
    def initialize_embeddings(self) -> None:
        """Initialize the embedding model"""
        try:
            self.logger.info(f"Initializing embeddings with model: {self.config.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            self.logger.info("Embeddings initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {str(e)}")
            raise

    @timer_decorator
    def load_existing_vectorstore(self, persist_directory: str, prompt_template: Optional[PromptTemplate] = None) -> None:
        """Load existing vectorstore from disk using Chroma"""
        try:
            self.logger.info(f"Loading existing vectorstore from {persist_directory}")
            
            # Initialize embeddings if not already done
            if not self.embeddings:
                self.initialize_embeddings()
            
            # Get or create collection using the persistent client
            self.vectorstores['similarity'] = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            
            # Setup retrievers
            self.setup_retriever()
            
            # Initialize LLM if not already done
            if not self.llm:
                self.initialize_llm()
            
            # Setup chain if prompt template provided
            if prompt_template:
                self.setup_chain(prompt_template)
            
            self.logger.info("Successfully loaded existing vectorstore")
            
        except Exception as e:
            self.logger.error(f"Error loading existing vectorstore: {str(e)}")
            raise

    @timer_decorator
    def create_vectorstore(self) -> None:
        """Create vector store from document splits with persistence"""
        try:
            if not self.splits:
                raise ValueError("No document splits available")
            if not self.embeddings:
                self.initialize_embeddings()

            self.logger.info("Creating vector store")
            
            # Create new collection and vectorstore
            self.vectorstores['similarity'] = Chroma.from_documents(
                documents=self.splits,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                client=self.chroma_client
            )
            
            self.logger.info("Vector store created successfully")

        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    @timer_decorator
    def _setup_similarity_retriever(self, retriever_config: RetrieverConfig) -> None:
        """Set up a similarity-based retriever"""
        try:
            self.logger.info("Setting up similarity-based retriever")
            if not self.vectorstores.get('similarity'):
                if not self.splits:
                    raise ValueError("No document splits available")
                if not self.embeddings:
                    self.initialize_embeddings()
                
                self.vectorstores['similarity'] = Chroma.from_documents(
                    self.splits,
                    self.embeddings
                )
            
            retriever = self.vectorstores['similarity'].as_retriever(
                search_kwargs={"k": retriever_config.top_k}
            ).configurable_fields(
                search_kwargs=ConfigurableField(
                    id="search_kwargs_similarity",
                    name="Search Kwargs",
                    description="The search kwargs to use for similarity search"
                )
            )
            self.retrievers['similarity'] = retriever
            self.logger.info("Similarity-based retriever setup successfully")
            return retriever

        except Exception as e:
            self.logger.error(f"Error setting up similarity retriever: {str(e)}")
            raise

    @timer_decorator
    def _setup_bm25_retriever(self, retriever_config: RetrieverConfig) -> None:
        """Set up a BM25 retriever"""
        try:
            self.logger.info("Setting up BM25 retriever")
            if not self.splits:
                raise ValueError("No document splits available")
            
            texts = [doc.page_content for doc in self.splits]
            metadatas = [doc.metadata for doc in self.splits]
            
            retriever = BM25Retriever.from_texts(
                texts,
                metadatas=metadatas
            )
            retriever.k = retriever_config.top_k
            
            self.retrievers['bm25'] = retriever
            self.logger.info("BM25 retriever setup successfully")
            return retriever

        except Exception as e:
            self.logger.error(f"Error setting up BM25 retriever: {str(e)}")
            raise

    @timer_decorator
    def setup_retriever(self) -> None:
        """Set up the retriever(s) based on configuration"""
        try:
            self.logger.info("Setting up retrievers")
            
            # Initialize individual retrievers
            retriever_instances = []
            weights = []
            
            for retriever_config in self.config.retrievers:
                if retriever_config.retriever_type == RetrieverType.SIMILARITY:
                    retriever = self._setup_similarity_retriever(retriever_config)
                elif retriever_config.retriever_type == RetrieverType.BM25:
                    retriever = self._setup_bm25_retriever(retriever_config)
                else:
                    raise ValueError(f"Unsupported retriever type: {retriever_config.retriever_type}")
                
                retriever_instances.append(retriever)
                weights.append(retriever_config.weight)
            
            # Set up ensemble retriever if multiple retrievers are configured
            if len(retriever_instances) > 1:
                self.logger.info("Setting up ensemble retriever")
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=retriever_instances,
                    weights=weights
                )
                self.logger.info("Ensemble retriever setup successfully")
            else:
                self.ensemble_retriever = retriever_instances[0]
                
            self.logger.info("All retrievers setup successfully")

        except Exception as e:
            self.logger.error(f"Error setting up retrievers: {str(e)}")
            raise

    @timer_decorator
    def initialize_llm(self) -> None:
        """Initialize the language model"""
        try:
            if not os.getenv('GROQ_API_KEY'):
                raise ValueError("GROQ_API_KEY environment variable not set")

            self.logger.info(f"Initializing LLM with model: {self.config.llm_model}")
            self.llm = ChatGroq(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            self.logger.info("LLM initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing LLM: {str(e)}")
            raise

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format retrieved documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)

    @timer_decorator
    def setup_chain(self, prompt_template: PromptTemplate) -> None:
        """Set up the RAG chain"""
        try:
            if not self.ensemble_retriever or not self.llm:
                raise ValueError("Retriever and LLM must be initialized first")

            self.logger.info("Setting up RAG chain")
            self.chain = (
                {
                    "context": self.ensemble_retriever | self.format_docs,
                    "user_query": RunnablePassthrough()
                }
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            self.logger.info("RAG chain setup successfully")

        except Exception as e:
            self.logger.error(f"Error setting up chain: {str(e)}")
            raise

    @timer_decorator
    def initialize_full_pipeline(self, file_path: Union[str, Path], prompt_template: PromptTemplate) -> None:
        """Initialize the complete RAG pipeline"""
        try:
            self.logger.info("Initializing full RAG pipeline")
            self.load_document(file_path)
            self.split_text()
            self.initialize_embeddings()
            self.create_vectorstore()
            self.setup_retriever()
            self.initialize_llm()
            self.setup_chain(prompt_template)
            self.logger.info("Full RAG pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing full pipeline: {str(e)}")
            raise

    @timer_decorator
    def query(self, query: str) -> str:
        """Execute a query through the RAG pipeline with connection check"""
        try:
            if not self.chain:
                raise ValueError("RAG chain not initialized")
            
            # Check if collection is accessible
            if not self.chroma_client.get_collection(self.collection_name):
                raise ValueError("Collection not found. Reconnecting...")
            
            self.logger.info(f"Executing query: {query[:100]}...")
            response = self.chain.invoke(query)
            self.logger.info("Query executed successfully")
            return response

        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        try:
            self.logger.info(f"Updating configuration with parameters: {kwargs}")
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    raise ValueError(f"Invalid configuration parameter: {key}")
            self.logger.info("Configuration updated successfully")

        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            raise