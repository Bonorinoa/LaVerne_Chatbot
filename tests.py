from utils import RAGConfig, RAG, RetrieverConfig, RetrieverType

from langchain_core.prompts import PromptTemplate

import tempfile
from pathlib import Path

def create_sample_document():
    """Create a sample AI generated document for testing"""
    
    sample_text = """
    Machine Learning and Artificial Intelligence in Healthcare
    
    Machine learning (ML) and artificial intelligence (AI) are revolutionizing healthcare delivery. 
    These technologies are being used to improve diagnosis, treatment planning, and patient care.
    
    Key Applications:
    1. Disease Diagnosis: AI algorithms can analyze medical images to detect diseases early.
    2. Treatment Planning: ML models help doctors choose the most effective treatments.
    3. Patient Monitoring: AI systems can monitor patient vital signs in real-time.
    4. Drug Discovery: ML accelerates the process of discovering new medications.
    
    Challenges:
    - Data Privacy: Protecting sensitive patient information is crucial.
    - Integration: Healthcare systems must adapt to incorporate new technologies.
    - Training: Medical professionals need training to use AI tools effectively.
    - Cost: Implementation of AI systems can be expensive.
    
    Future Prospects:
    The future of healthcare will likely see increased automation and personalized medicine.
    AI will continue to enhance medical decision-making and improve patient outcomes.
    """
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / "healthcare_ai.txt"
    with open(file_path, "w") as f:
        f.write(sample_text)
    
    return str(file_path)

def compare_retrievers():
    """Compare single vs ensemble retrievers"""
    # Create sample document
    file_path = create_sample_document()
    
    # Prompt template
    prompt_template = PromptTemplate(
        template="""Based on the following context, answer the question concisely:
        
        Context: {context}
        
        Question: {user_query}
        
        Answer:""",
        input_variables=["context", "user_query"]
    )
    
    # Test queries
    queries = [
        "What are the main applications of AI in healthcare?",
        "What challenges exist in implementing AI systems?",
        "How does machine learning help in drug discovery?"
    ]
    
    # Test single retriever
    print("Testing Single Retriever (Similarity)...")
    single_config = RAGConfig(
        chunk_size=200,  # Smaller chunks for this example
        chunk_overlap=50,
        retrievers=[
            RetrieverConfig(
                retriever_type=RetrieverType.SIMILARITY,
                top_k=2,
                weight=1.0
            )
        ]
    )
    
    single_rag = RAG(single_config)
    single_rag.initialize_full_pipeline(file_path, prompt_template)
    
    # Test ensemble retriever
    print("\nTesting Ensemble Retriever (Similarity + BM25)...")
    ensemble_config = RAGConfig(
        chunk_size=200,
        chunk_overlap=50,
        retrievers=[
            RetrieverConfig(
                retriever_type=RetrieverType.SIMILARITY,
                top_k=2,
                weight=0.5
            ),
            RetrieverConfig(
                retriever_type=RetrieverType.BM25,
                top_k=2,
                weight=0.5
            )
        ]
    )
    
    ensemble_rag = RAG(ensemble_config)
    ensemble_rag.initialize_full_pipeline(file_path, prompt_template)
    
    # Compare responses
    print("\nComparing Responses:")
    print("-" * 80)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        single_response = single_rag.query(query)
        ensemble_response = ensemble_rag.query(query)
        
        print("Single Retriever Response:")
        print(single_response)
        print("\nEnsemble Retriever Response:")
        print(ensemble_response)
        print("-" * 80)

if __name__ == "__main__":
    compare_retrievers()