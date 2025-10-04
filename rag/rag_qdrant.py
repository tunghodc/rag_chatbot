from typing import Generator
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from config import DB_DIR, DB_COLLECTION, OPENAI_API_KEY, OPENAI_API_BASE, LLM_MODEL, SEARCH_TYPE
from vectorstore.qdrant_store import get_qdrant_vectorstore
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
        self.vector_store = get_qdrant_vectorstore(collection_name=DB_COLLECTION, search_type=SEARCH_TYPE)

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    def query_streaming(self, question: str) -> Generator[str, None, None]:
        """Query with streaming response"""
    
        try:
            # Get relevant documents first
            relevant_docs = self.vector_store.similarity_search(question) if self.vector_store else []
            print('Relevant docs:', relevant_docs)
            
            # Format context
            context = self._format_docs(relevant_docs)
            print('Context:', context)
            
            # Create prompt
            prompt = self.prompt.format_messages(context=context, question=question)
            print('Prompt:', prompt)
            
            # Stream tokens
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
                    
        except Exception as e:
            yield f"Error: {str(e)}"
