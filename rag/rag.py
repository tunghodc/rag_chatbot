from typing import Generator
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from config import CHROMA_DIR, CHROMA_COLLECTION, OPENAI_API_KEY, OPENAI_API_BASE, LLM_MODEL
from vectorstore.chroma_store import get_langchain_chroma_retriever, get_langchain_chroma_vectorstore

class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated"""
        self.tokens.append(token)

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with document upload and streaming"""
    
    def __init__(self):     
        # Initialize components
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Set up the enhanced RAG pipeline"""   
        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_API_BASE,
                streaming=True,
                temperature=0.0,
            )

        prompt_template = """You are a helpful AI assistant with access to a knowledge base.
        Use the following context to answer the question.
        If you cannot find the answer in the context, say you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""

        self.prompt = ChatPromptTemplate.from_template(prompt_template)

        # Initialize or load vector store
        self._initialize_retriever()
        
    
    def _initialize_retriever(self):
        """Load the vector store"""
        self.vector_store = get_langchain_chroma_vectorstore(persist_directory=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
        self.retriever = get_langchain_chroma_retriever(persist_directory=CHROMA_DIR, collection_name=CHROMA_COLLECTION)
        
        # Create retriever
        if not self.retriever:
            raise ValueError("Failed to initialize retriever from ChromaDB.")


    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    

    
    def query(self, question: str) -> str:
        """Query the RAG pipeline"""
        
        self.rag_chain = (
                {'context': self.retriever | self._format_docs, 'question': RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        try:
            result = self.rag_chain.invoke(question)
            if result:
                return result
            else:
                return "No answer found in the knowledge base."
            
        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"
    
    def query_streaming(self, question: str) -> Generator[str, None, None]:
        """Query with streaming response"""
    
        try:
            # Set streaming callback
            streaming_handler = StreamingCallbackHandler()
            self.llm.callbacks = [streaming_handler]
            
            # Get relevant documents first
            relevant_docs = self.retriever.invoke(question) if self.retriever else []
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Stream the response
            prompt = self.prompt.format_messages(context=context, question=question)
            
            # Stream tokens
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
                    
        except Exception as e:
            yield f"Error: {str(e)}"
