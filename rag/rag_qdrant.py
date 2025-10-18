from typing import Generator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import torch
from config import DB_COLLECTION, OPENAI_API_KEY, OPENAI_API_BASE, LLM_MODEL, SEARCH_TYPE, POST_RANKING
from vectorstore.qdrant_store import get_qdrant_vectorstore
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Reranker")
rerank_model = AutoModelForSequenceClassification.from_pretrained("AITeamVN/Vietnamese_Reranker").to(device)
rerank_model.eval()

def rerank_fn(query, docs):
    # docs: list of Document objects with .page_content or .content attributes
    
    with torch.no_grad():
        inputs = tokenizer(
        [(query, doc.page_content) for doc in docs],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=1256,
    )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    # Return docs sorted by score
    scored = list(zip(docs, scores.tolist()))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    print("SCORED", scored_sorted)
    return [doc for doc, score in scored_sorted]

class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated"""
        self.tokens.append(token)

class RAGPipeline:
    """RAG pipeline with document upload and streaming"""
    
    def __init__(self):     
        # Initialize components
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Set up the RAG pipeline"""   
        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE,
                streaming=True,
                callbacks=[StreamingCallbackHandler()],
                temperature=0.0,
            )

        prompt_template = """You are a helpful AI assistant with access to a knowledge base.
        Use the following context to answer the question.
        If you cannot find the answer in the context, say you don't know. Response in Vietnamese.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""

        self.prompt = ChatPromptTemplate.from_template(prompt_template)

        # Initialize or load vector store
        self._initialize_vector_store()
        
    
    def _initialize_vector_store(self):
        """Load the vector store"""
        self.vector_store_dense = get_qdrant_vectorstore(collection_name=DB_COLLECTION, search_type="dense")
        self.vector_store_sparse = get_qdrant_vectorstore(collection_name=DB_COLLECTION, search_type="sparse")
        self.vector_store_hybrid = get_qdrant_vectorstore(collection_name=DB_COLLECTION, search_type="hybrid")
        self.vector_store = get_qdrant_vectorstore(collection_name=DB_COLLECTION, search_type=SEARCH_TYPE)

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    def query_streaming(self, question: str) -> Generator[str, None, None]:
        """Query with streaming response"""
        try:
            # Get relevant documents first
            if SEARCH_TYPE == "hybrid":
                if POST_RANKING == "rrf":
                    relevant_docs = self.vector_store_hybrid.similarity_search(question)
                elif POST_RANKING == "rerank":
                    print('Using reranking...')
                    dense_results = self.vector_store_dense.similarity_search(question, k=6)
                    sparse_results = self.vector_store_sparse.similarity_search(question, k=6)
                    merged = {}
                    for doc in dense_results + sparse_results:
                        doc_id = doc.metadata.get("_id")
                        merged[doc_id] = doc  # later ones overwrite earlier ones
                    relevant_docs = list(merged.values())
                    print('Combined results before reranking:', len(relevant_docs))
            else:
                relevant_docs = self.vector_store.similarity_search(question, k=3)

            # Rerank the retrieved docs
            if relevant_docs:
                reranked = rerank_fn(question, relevant_docs)
                relevant_docs = reranked[:3]
            print('Final relevant docs after reranking:', len(relevant_docs))
            # print('Relevant docs:', relevant_docs)
            # Format context
            context = self._format_docs(relevant_docs)
            # print('Context:', context)
            
            # Create prompt
            prompt = self.prompt.format_messages(context=context, question=question)
            # print('Prompt:', prompt)
            
            # Stream tokens
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
                    
        except Exception as e:
            yield f"Error: {str(e)}"

if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    question = "Lợi ích của việc nghiên cứu cây lúa là gì?"
    print("Question:", question)
    print("Answer:")
    for token in rag_pipeline.query_streaming(question):
        print(token, end="", flush=True)    
