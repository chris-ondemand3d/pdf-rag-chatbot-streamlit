#Import Library
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda

from langchain_postgres.vectorstores import PGVector
from database import COLLECTION_NAME, CONNECTION_STRING
from langchain_community.storage import RedisStore
from langchain.schema.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path
from base64 import b64decode
import os, hashlib, shutil, uuid, json, time
import torch, redis, streamlit as st
import logging


from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

POPPLER_PATH = os.path.join(os.path.dirname(__file__), "poppler", "poppler-24.08.0", "Library", "bin")
os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

# Ensure PyTorch module path is correctly set
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Redis client
client = redis.Redis(host="localhost", port=6379, db=0)




#Data Loading
def load_pdf_data(file_path):
    logging.info(f"Data ready to be partitioned and loaded ")
    raw_pdf_elements = partition_pdf(
        filename=file_path,
      
        infer_table_structure=True,
        strategy="fast",

        extract_image_block_types = ["Image"],
        extract_image_block_to_payload  = True,

        chunking_strategy="by_title",     
        mode='elements',
        max_characters=10000,
        new_after_n_chars=5000,
        combine_text_under_n_chars=2000,
        image_output_dir_path="data/",
    )
    logging.info(f"Pdf data finish loading, chunks now available!")
    return raw_pdf_elements

# Generate a unique hash for a PDF file
def get_pdf_hash(pdf_path):
    """Generate a SHA-256 hash of the PDF file content."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return hashlib.sha256(pdf_bytes).hexdigest()

# Summarize extracted text and tables using LLM
def summarize_text_and_tables(text, tables, status_text=None):
    logging.info("Ready to summarize data with LLM")
    prompt_text = """You are an assistant tasked with summarizing text and tables. \

                    You are to give a concise summary of the table or text and do nothing else.
                    Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(temperature=0.6, model="gemini-3-flash-preview")
    summarize_chain = (
        {"element": RunnablePassthrough()}
        | prompt
        | model.with_retry(stop_after_attempt=5, wait_exponential_jitter=True)
        | StrOutputParser()
    )

    def summarize_with_progress(items, label):
        results = []
        for i, item in enumerate(items):
            if status_text:
                status_text.write(f"📝 Summarizing {label} {i+1}/{len(items)}...")
            logging.info(f"Summarizing {label} {i+1}/{len(items)}")
            results.append(summarize_chain.invoke(item))
            time.sleep(1)
        return results

    return {
        "text": summarize_with_progress(text, "text chunk"),
        "table": summarize_with_progress(tables, "table"),
    }
  
#Initialize a pgvector and retriever for storing and searching documents
def initialize_retriever():

    store = RedisStore(client=client)
    id_key = "doc_id"
    vectorstore = PGVector(
            embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
            )
    retrieval_loader = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    return retrieval_loader


# Store text, tables, and their summaries in the retriever

def store_docs_in_retriever(text, text_summary, table, table_summary, retriever):
    """Store text and table documents along with their summaries in the retriever."""

    def add_documents_to_retriever(documents, summaries, retriever, id_key = "doc_id"):
        """Helper function to add documents and their summaries to the retriever."""
        if not summaries:
            return None, []

        doc_ids = [str(uuid.uuid4()) for _ in documents]
        summary_docs = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs, ids=doc_ids)
        retriever.docstore.mset(list(zip(doc_ids, documents)))     

# Add text, table, and image summaries to the retriever
    add_documents_to_retriever(text, text_summary, retriever)
    add_documents_to_retriever(table, table_summary, retriever)
    return retriever


# Parse the retriever output
def parse_retriver_output(data):
    parsed_elements = []
    for element in data:
        # Decode bytes to string if necessary
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        
        parsed_elements.append(element)
    
    return parsed_elements


# Chat with the LLM using retrieved context

def chat_with_llm(retriever):

    logging.info(f"Context ready to send to LLM ")
    prompt_text = """
                You are an AI Assistant tasked with understanding detailed
                information from text and tables. You are to answer the question based on the 
                context provided to you. You must not go beyond the context given to you.
                
                Context:
                {context}

                Question:
                {question}
                """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGoogleGenerativeAI(temperature=0.6, model="gemini-3-flash-preview")
 
    rag_chain = ({
       "context": retriever | RunnableLambda(parse_retriver_output), "question": RunnablePassthrough(),
        } 
        | prompt 
        | model 
        | StrOutputParser()
        )
        
    logging.info(f"Completed! ")

    return rag_chain

# Generate temporary file path of uploaded docs
def _get_file_path(file_upload):

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    if isinstance(file_upload, str):
        file_path = file_upload  # Already a string path
    else:
        file_path = os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        return file_path
    

# Process uploaded PDF file with incremental progress updates
def process_pdf(file_upload, progress_bar=None, status_text=None):

    def update(step, total, msg):
        if progress_bar:
            progress_bar.progress(step / total, text=msg)
        if status_text:
            status_text.write(msg)
        logging.info(msg)

    update(1, 6, "📄 Saving file...")
    file_path = _get_file_path(file_upload)
    pdf_hash = get_pdf_hash(file_path)

    update(2, 6, "🔌 Connecting to vector store...")
    load_retriever = initialize_retriever()

    if client.exists(f"pdf:{pdf_hash}"):
        update(6, 6, "✅ PDF already indexed — ready to chat!")
        return load_retriever

    update(3, 6, "🔍 Parsing PDF into chunks...")
    pdf_elements = load_pdf_data(file_path)

    tables = [element.metadata.text_as_html for element in
               pdf_elements if 'Table' in str(type(element))]
    text = [element.text for element in pdf_elements if
            'CompositeElement' in str(type(element))]

    update(4, 6, f"📝 Summarizing {len(text)} text chunks and {len(tables)} tables...")
    summaries = summarize_text_and_tables(text, tables, status_text)

    update(5, 6, "💾 Indexing embeddings in vector store...")
    retriever = store_docs_in_retriever(text, summaries['text'], tables, summaries['table'], load_retriever)

    filename = file_upload.name if hasattr(file_upload, "name") else os.path.basename(file_upload)
    client.set(f"pdf:{pdf_hash}", json.dumps({"text": "PDF processed", "filename": filename}))
    update(6, 6, "✅ PDF fully indexed — ready to chat!")
    return retriever


#Invoke chat with LLM based on uploaded PDF and user query
def invoke_chat(file_upload, message):

    retriever = st.session_state.get("retriever") or process_pdf(file_upload)
    rag_chain = chat_with_llm(retriever)
    response = rag_chain.invoke(message)
    response_placeholder = st.empty()
    response_placeholder.write(response)
    return response


# Main application interface using Streamlit
def main():
  
    st.title("PDF Chat Assistant ")
    logging.info("App started")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

 
    st.sidebar.markdown("### Uploaded PDFs")
    pdf_keys = client.keys("pdf:*")
    if pdf_keys:
        for k in pdf_keys:
            data = client.get(k)
            if data:
                info = json.loads(data)
                filename = info.get("filename", "Unknown")
                st.sidebar.markdown(f"- 📄 {filename}")
    else:
        st.sidebar.caption("No PDFs uploaded yet.")

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
        with st.sidebar:
            with st.spinner("Clearing Redis and PGVector..."):
                client.flushdb()
                retriever = initialize_retriever()
                retriever.vectorstore.delete_collection()
            st.session_state["retriever"] = None
            st.session_state["processed_file_id"] = None
            st.sidebar.success("✅ All data cleared.")
            st.rerun()

    st.sidebar.markdown("---")
    file_upload = st.sidebar.file_uploader(
        label="Upload PDF", type=["pdf"],
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    if file_upload:
        file_id = f"{file_upload.name}_{file_upload.size}"
        if st.session_state.get("processed_file_id") != file_id:
            st.session_state["processed_file_id"] = file_id
            st.session_state["retriever"] = None
            with st.sidebar:
                status_text = st.empty()
                progress_bar = st.progress(0, text="Starting...")
                try:
                    retriever = process_pdf(file_upload, progress_bar, status_text)
                    st.session_state["retriever"] = retriever
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

            # if st.session_state.get("retriever"):
            #     with st.chat_message("assistant"):
            #         with st.spinner("Generating document summary..."):
            #             rag_chain = chat_with_llm(st.session_state["retriever"])
            #             summary = rag_chain.invoke("What is this document about?")
            #         st.session_state.messages.append({"role": "assistant", "content": summary})
            #         st.write(summary)
        else:
            st.sidebar.success("✅ PDF ready — ask your question!")

    # Prompt for user input
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Generate response if last message is not from assistant
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            start_time = time.time()
            logging.info("Generating response...")
            with st.spinner("Processing..."):
                user_message = " ".join([msg["content"] for msg in st.session_state.messages if msg])           
                response_message = invoke_chat(file_upload, user_message)
        
                duration = time.time() - start_time
                response_msg_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"

                st.session_state.messages.append({"role": "assistant", "content": response_msg_with_duration})
                st.write(f"Duration: {duration:.2f} seconds")
                logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")


    



if __name__ == "__main__":
    main()