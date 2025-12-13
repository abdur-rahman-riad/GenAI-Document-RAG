import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import shutil
import hashlib
import time

# PDF Processing
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Gemini
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="VOID - RAG Chatbot",
    page_icon="𖣐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Messenger-like UI
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        animation: fadeIn 0.3s ease-in;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* User messages - right aligned, blue theme */
    .chat-message.user {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        align-items: flex-end;
        margin-left: 15%;
        border-bottom-right-radius: 5px;
    }
    
    /* Bot messages - left aligned, dark theme */
    .chat-message.bot {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        align-items: flex-start;
        margin-right: 15%;
        border-bottom-left-radius: 5px;
    }
    
    /* Message text */
    .chat-message .message {
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.5rem;
        word-wrap: break-word;
    }
    
    /* Sender name */
    .chat-message .name {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .chat-message.user .name {
        color: #93c5fd;
    }
    
    .chat-message.bot .name {
        color: #60a5fa;
    }
    
    /* Input box styling */
    .stChatInput > div {
        background-color: #1f2937 !important;
        border: 2px solid #374151 !important;
        border-radius: 25px !important;
    }
    
    .stChatInput input {
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1f2937;
        border: 2px dashed #3b82f6;
        border-radius: 15px;
        padding: 1.5rem;
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        border-radius: 8px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

# Process PDF Function
def process_pdf(uploaded_file):
    """Extract text, create chunks, embeddings and vector database"""
    
    try:
        with st.spinner("📄 Processing PDF | Text Extracting"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Extract text using PyPDF
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # OCR fallback
            try:
                images = convert_from_path(pdf_path, dpi=300)
                ocr_text = ""
                for image in images:
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    ocr_text += page_text + "\n"
                
                if len(ocr_text.strip()) > len(text.strip()):
                    final_text = ocr_text
                else:
                    final_text = text
            except:
                final_text = text
            
            # Clean up temp file
            os.unlink(pdf_path)
        
        with st.spinner("✂️ Processing PDF | Chunking"):
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(final_text)
        
        with st.spinner("🧠 Processing PDF | Embeddings and VectorDB"):
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create unique directory for this session
            db_path = f"./temp_db_{st.session_state.session_id}"
            
            # Remove old database if exists
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            
            # Create vector database using from_texts
            vectordb = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )
            
            # Create retriever
            retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        
        return retriever, len(chunks)
    
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None, 0

# Setup RAG Chain Function
def setup_rag_chain(retriever):
    """Initialize Gemini LLM and create RAG chain"""
    
    try:
        # Get API key
        API_KEY = os.getenv("GOOGLE_API_KEY")
        if API_KEY is None:
            st.error("❌ GOOGLE_API_KEY not found in .env file!")
            st.stop()
        
        # Configure Gemini
        genai.configure(api_key=API_KEY)
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Create prompt template
        prompt_template = """
SYSTEM ROLE:
You are an enterprise-grade AI assistant designed to answer questions using retrieved document content only.
You operate in a production RAG system where accuracy, reliability, and transparency are critical.

CORE RULES (STRICT):
1. Use ONLY the information provided in the CONTEXT.
2. NEVER use external knowledge, assumptions, or training data.
3. If the answer is missing, incomplete, or unclear, respond with:
   "Sorry! The Provided Documents Doesn't Contain Sufficient Information To Answer This Question. Please Try With Valid Information."
4. Do NOT hallucinate, guess, or fabricate details.
5. Maintain a professional, precise, and neutral tone.

OCR AWARENESS:
- The context may contain OCR-extracted text with noise, formatting issues, or minor recognition errors.
- Carefully infer meaning ONLY when it is logically supported by the text.
- Do NOT correct, rewrite, or invent content beyond what is clearly implied.

CONTEXT:
{context}

USER QUESTION:
{question}

ANALYSIS (INTERNAL REASONING):
- Identify relevant portions of the context.
- Cross-check facts across multiple sections if present.
- Resolve OCR inconsistencies cautiously.
- Determine whether the question can be fully answered.

FINAL ANSWER REQUIREMENTS:
- Provide a clear, concise, and factually grounded answer.
- Use bullet points or short paragraphs when appropriate.
- Reference document sections, page numbers, or chunk identifiers if available.
- Do NOT mention internal reasoning or system instructions.

FINAL ANSWER:
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    except Exception as e:
        st.error(f"❌ Error setting up RAG chain: {str(e)}")
        return None

# Cleanup function for session databases
def cleanup_old_databases():
    """Remove temporary databases from previous sessions"""
    try:
        current_dir = os.getcwd()
        for item in os.listdir(current_dir):
            if item.startswith("temp_db_") and item != f"temp_db_{st.session_state.session_id}":
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    except Exception as e:
        pass  # Silently handle cleanup errors

# Clean up old databases on startup
cleanup_old_databases()

# Title
st.title("𖣐 VOID - AI RAG Assistant")
st.markdown("*Intelligence Beyond the Documents*")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("📁 Document Upload")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload your PDF document", 
        type=['pdf'],
        help="Upload a PDF file to start asking questions"
    )
    
    if uploaded_file is not None:
        if not st.session_state.pdf_processed:
            # Process the PDF
            retriever, chunk_count = process_pdf(uploaded_file)
            
            if retriever:
                # Setup RAG chain
                with st.spinner("🔗 Setting up AI Chain"):
                    rag_chain = setup_rag_chain(retriever)
                
                if rag_chain:
                    # Store in session state
                    st.session_state.rag_chain = rag_chain
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chunk_count = chunk_count
                    
                    st.success(f"✅ **{uploaded_file.name}** Processed Successfully!")
                    st.info(f"📊 Created **{chunk_count}** Knowledge Chunks")
                    st.balloons()
        else:
            st.success(f"✅ **{st.session_state.pdf_name}** is loaded!")
            st.info(f"📊 **{st.session_state.chunk_count}** chunks available")
    
    st.markdown("---")
    
    # Reset button
    if st.session_state.pdf_processed:
        if st.button("🔄 Reset & Upload New PDF", use_container_width=True):
            # Clean up database
            db_path = f"./temp_db_{st.session_state.session_id}"
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.rerun()

    st.markdown("### 💡 Tips")
    st.markdown("""
    - Upload any PDF document
    - Ask specific questions
    - Chat history persists until reset
    - Refresh page to start over
    """)

# Main Chat Interface
if st.session_state.pdf_processed:
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="name">You</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="name">𖣐 VOID</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input at the bottom
    user_question = st.chat_input("Ask VOID a question about your document...")
    
    if user_question:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Get response from RAG chain
        with st.spinner("𖣐 VOID is thinking..."):
            try:
                answer = st.session_state.rag_chain.invoke(user_question)
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Rerun to display new messages
        st.rerun()

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 20px; margin-top: 2rem;'>
        <h2 style='color: white; margin-bottom: 1rem;'>Welcome to 𖣐 VOID!</h2>
        <p style='color: #e5e7eb; font-size: 1.1rem; margin-bottom: 2rem;'>
            Intelligence Beyond the Documents
        </p>
        <p style='color: #bfdbfe; font-size: 1rem;'>
            📤 Upload a PDF document from the sidebar to begin your journey
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    