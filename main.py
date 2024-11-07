from utils import RAGConfig, RAG, RetrieverConfig, RetrieverType

from langchain_core.prompts import PromptTemplate

# test
# Create configuration
config = RAGConfig(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="llama3-8b-8192",
    temperature=0.0,
    max_tokens=200,
    retrievers=[
        RetrieverConfig(RetrieverType.SIMILARITY, top_k=4, weight=0.5),
        RetrieverConfig(RetrieverType.BM25, top_k=4, weight=0.5)
    ]
)

# Create prompt template
prompt = PromptTemplate.from_template(
    """Answer the question based on the following context:
    
    Context: {context}
    
    Question: {user_query}
    
    Answer: """
)

# Initialize RAG
rag = RAG(config)

# Initialize full pipeline
rag.initialize_full_pipeline("ulv-faculty-handbook 23-24.pdf", prompt)

# Execute query
response = rag.query("where is the Title IX office located?")

print(response)