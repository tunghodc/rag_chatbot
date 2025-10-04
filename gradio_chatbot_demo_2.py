import gradio as gr
from pathlib import Path
from rag.rag_qdrant import RAGPipeline

from ingest.ingest_pdfs import ingest_pdf
from ingest.ingest_urls import ingest_urls

# Initialize components
rag_pipeline = RAGPipeline()

# Global state for uploaded documents
uploaded_docs_info = {"count": 0, "names": []}

def process_uploaded_file(file) -> str:
    """Process uploaded document and add to RAG pipeline"""
    global uploaded_docs_info
    if file is None:
        return "No file uploaded."

    try:
        file_path = file.name
        file_extension = Path(file_path).suffix.lower()
        if file_path in uploaded_docs_info["names"]:
            return f"File '{Path(file_path).name}' already uploaded."
        
        # Select appropriate loader
        if file_extension == '.pdf':
            ingest_pdf(file_path)
        elif file_extension == '.txt':
            ingest_urls(file_path)  # Assuming text files contain URLs
        else:
            return f"Unsupported file type: {file_extension}"

        # Update global state
        uploaded_docs_info["count"] += 1
        uploaded_docs_info["names"].append(Path(file_path).name)

        return f"✅ Successfully processed '{Path(file_path).name}' added to knowledge base"
        
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"


def chat_interface_streaming(message, history):
    """chat interface with streaming support"""
    if not message:
        return "", history
    
    try:
        # Streaming response
        print("Processing message with streaming...", history)
        history = history or []
        history.append([message, ""])
        
        # Get streaming response
        response_generator = rag_pipeline.query_streaming(message)
        
        full_response = ""
        for chunk in response_generator:
            full_response += chunk
            history[-1][1] = full_response
            yield "", history
            
    except Exception as e:
        error_msg = f"Xin lỗi, tôi đang gặp lỗi: {str(e)}"
        history = history or []
        history.append([message, error_msg])
        return "", history


def get_rag_info() -> str:
    """Get information about current RAG pipeline state"""
    info = f"""**RAG Pipeline Status:**
    
- **Documents Uploaded:** {uploaded_docs_info['count']}
- **Document Names:** {', '.join(uploaded_docs_info['names']) if uploaded_docs_info['names'] else 'None'}
"""
    return info

def search_knowledge_base(query: str, k: int = 3) -> str:
        """Search for similar documents"""

        print("Searching for similar documents...")
        try:
            results = rag_pipeline.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                source = doc.metadata.get("doc_title", "Unknown")
                content = doc.page_content
                formatted_results.append(
                    f"**Source:** {source}  |  **Similarity:** {score}\n\n{content}\n"
                )
            
            formatted_str = "🔍 **Search Results:**\n\n"
            for i, doc in enumerate(formatted_results, 1):
                formatted_str += f"**Result {i}:**\n"
                formatted_str += f"\n{doc}\n\n\n"

            return formatted_str
            
        except Exception as e:
            return f"Search error: {str(e)}"


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Chatbot hỗ trợ nông nghiệp") as demo:
    
    with gr.Tabs():
        # Tab 1: Chat Interface
        with gr.TabItem("💬 Chatbot"):
            with gr.Column():
                chatbot = gr.Chatbot(
                    height=600,
                    bubble_full_width=False,
                    show_label=True
                )
                
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Đặt câu hỏi về nông nghiệp của bạn ở đây.",
                    lines=2
                )
                
                with gr.Row():
                    submit = gr.Button("Gửi", variant="primary")           
        
        #Tab 2: Knowledge Base Search
        with gr.TabItem("🔍 Knowledge Search"):
            gr.Markdown("### Direct Similarity Search in Knowledge Base")
            
            with gr.Row():
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter keywords or questions to search...",
                    lines=3,
                    scale=3
                )
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of Results",
                    scale=1
                )
            
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Markdown()

        # Tab 3: Document Upload
        with gr.TabItem("📄 Document Upload"):
            gr.Markdown("### Upload Documents for Q&A")
            
            # with gr.Row():
            file_upload = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".txt"],
                    type="filepath"
                )
                
            upload_btn = gr.Button("Process Document", variant="secondary")
            upload_status = gr.Textbox(
                label="Upload Status",
                lines=2,
                interactive=False
            )
            
            rag_info = gr.Markdown(get_rag_info())   
    

    # Event handlers
    def handle_upload(file):
        status = process_uploaded_file(file)
        return status, get_rag_info()
    
    # Chat events
    msg.submit(
        chat_interface_streaming,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    submit.click(
        chat_interface_streaming,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    # Upload events
    upload_btn.click(
        handle_upload,
        inputs=[file_upload],
        outputs=[upload_status, rag_info]
    )

    search_btn.click(
        search_knowledge_base,
        inputs=[search_query, num_results],
        outputs=search_results
    )


if __name__ == "__main__":
    print("\nStarting Gradio Chatbot Demo...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
