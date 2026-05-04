#Import Library
from umls_client import map_entities_to_umls, parse_entity_terms
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch, redis, streamlit as st
import logging


from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

POPPLER_PATH = os.path.join(os.path.dirname(__file__), "poppler", "poppler-24.08.0", "Library", "bin")
os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

# Ensure PyTorch module path is correctly set
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Configure logging — console + file
os.makedirs("logs", exist_ok=True)
_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
_logger.addHandler(_console_handler)

_file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
_file_handler.setFormatter(_log_formatter)
_logger.addHandler(_file_handler)

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
def summarize_text_and_tables(text, tables, status_text=None, progress_bar=None):
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

    total_items = len(text) + len(tables)
    overall_done = [0]

    def summarize_with_progress(items, max_workers=8):
        if not items:
            return []
        results = [None] * len(items)

        def run(i, item):
            return i, summarize_chain.invoke(item)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run, i, item): i for i, item in enumerate(items)}
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result
                overall_done[0] += 1
                msg = f"📝 Summarizing chunks: {overall_done[0]}/{total_items}"
                logging.info(msg)
                if status_text:
                    status_text.write(msg)
                if progress_bar:
                    sub = overall_done[0] / total_items if total_items > 0 else 1
                    progress_bar.progress((3 + sub) / 6, text=msg)

        return results

    return {
        "text": summarize_with_progress(text),
        "table": summarize_with_progress(tables),
    }
  
#Initialize a pgvector and retriever for storing and searching documents
def initialize_retriever(filename_filter=None):
    store = RedisStore(client=client)
    vectorstore = PGVector(
        embeddings=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    search_kwargs = {"filter": {"filename": filename_filter}} if filename_filter else {}
    retrieval_loader = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=store, id_key="doc_id",
        search_kwargs=search_kwargs,
    )
    return retrieval_loader


# Store text, tables, and their summaries in the retriever

def store_docs_in_retriever(text, text_summary, table, table_summary, retriever,
                             text_meta=None, table_meta=None):
    """Store text and table documents along with their summaries in the retriever."""

    def add_documents_to_retriever(documents, summaries, retriever, meta_list=None, id_key="doc_id"):
        """Helper function to add documents and their summaries to the retriever."""
        if not summaries:
            return None, []

        doc_ids = [str(uuid.uuid4()) for _ in documents]
        summary_docs = [
            Document(
                page_content=summary,
                metadata={
                    id_key: doc_ids[i],
                    **(meta_list[i] if meta_list else {}),
                }
            )
            for i, summary in enumerate(summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs, ids=doc_ids)
        retriever.docstore.mset(list(zip(doc_ids, documents)))

    add_documents_to_retriever(text, text_summary, retriever, meta_list=text_meta)
    add_documents_to_retriever(table, table_summary, retriever, meta_list=table_meta)
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


# Extract summary, table of contents, and entities for a specific PDF
def get_document_info(filename):
    retriever = initialize_retriever(filename_filter=filename)
    rag_chain = chat_with_llm(retriever)
    return {
        "summary": rag_chain.invoke(
            "Provide a comprehensive summary of this document in 3-5 sentences."
        ),
        "toc": rag_chain.invoke(
            "List the main sections, chapters, or topics covered in this document as a numbered table of contents."
        ),
        "entities": rag_chain.invoke(
            "Extract and list the key terms from this document, grouped into these categories: Anatomical Terms, Materials & Substances, Clinical Terms, and Technical Terms. Format as a grouped bullet list under each category heading."
        ),
    }


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

    filename = file_upload.name if hasattr(file_upload, "name") else os.path.basename(file_upload)

    tables = []
    tables_meta = []
    for element in pdf_elements:
        if 'Table' in str(type(element)):
            tables.append(element.metadata.text_as_html)
            tables_meta.append({
                "filename": filename,
                "page_number": getattr(element.metadata, "page_number", None),
                "chunk_type": "table",
            })

    text = []
    text_meta = []
    for element in pdf_elements:
        if 'CompositeElement' in str(type(element)):
            text.append(element.text)
            text_meta.append({
                "filename": filename,
                "page_number": getattr(element.metadata, "page_number", None),
                "chunk_type": "text",
            })

    update(4, 6, f"📝 Summarizing {len(text)} text chunks and {len(tables)} tables...")
    summaries = summarize_text_and_tables(text, tables, status_text, progress_bar)

    update(5, 6, "💾 Indexing embeddings in vector store...")
    retriever = store_docs_in_retriever(
        text, summaries['text'], tables, summaries['table'], load_retriever,
        text_meta=text_meta, table_meta=tables_meta
    )
    client.set(f"pdf:{pdf_hash}", json.dumps({"text": "PDF processed", "filename": filename}))
    update(6, 6, "✅ PDF fully indexed — ready to chat!")
    return retriever


def _render_umls_results(results: list):
    import pandas as pd

    mapped  = sum(1 for r in results if r["status"] == "mapped")
    partial = sum(1 for r in results if r["status"] == "partial")
    new     = sum(1 for r in results if r["status"] == "new")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(results))
    c2.metric("✅ Mapped",  mapped)
    c3.metric("⚠️ Partial", partial)
    c4.metric("🆕 New",     new)

    rows = []
    for r in results:
        icon   = "✅" if r["status"] == "mapped" else "⚠️" if r["status"] == "partial" else "🆕"
        snomed = "; ".join(f"{s['code']} {s['name']}" for s in r.get("snomed", []))
        parent = ""
        if r["status"] == "new":
            p = r.get("suggested_parent", {})
            pid = f" ({p['parent_id']})" if p.get("parent_id") else ""
            parent = f"{p.get('parent_name', '')}{pid}"
        rows.append({
            "":              icon,
            "Term":          r["term"],
            "UMLS Match":    r.get("name") or "",
            "CUI":           r.get("cui")  or "",
            "Similarity":    r.get("similarity") or "",
            "Semantic Types":  ", ".join(r.get("semantic_types", [])),
            "SNOMED-CT":     snomed,
            "Suggested Parent": parent,
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "":              st.column_config.TextColumn("", width="small"),
            "Similarity":    st.column_config.NumberColumn(format="%.3f"),
        },
    )


def extract_relationships(entities: list[str], filename: str) -> list[dict]:
    """Ask the RAG chain to extract pairwise relationships between entities."""
    retriever = initialize_retriever(filename_filter=filename)
    rag_chain  = chat_with_llm(retriever)
    entity_list = "\n".join(f"- {e}" for e in entities[:30])
    raw = rag_chain.invoke(
        f"Based on the document, extract medical/clinical relationships between these entities:\n"
        f"{entity_list}\n\n"
        "Return ONLY a JSON array (no markdown). Each object must have:\n"
        '{"source": "...", "relation": "...", "target": "..."}\n'
        "Use concise relation labels: causes, treats, part_of, associated_with, used_for, affects, contains, etc.\n"
        "Return up to 30 relationships. Use only entities from the list above."
    )
    import re as _re, json as _json
    m = _re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            return _json.loads(m.group(0))
        except Exception:
            pass
    return []


def render_entity_graph(umls_results: list[dict], relationships: list[dict]):
    """Render an interactive force-directed entity-relationship graph using pyvis."""
    import tempfile, os
    from pyvis.network import Network
    import streamlit.components.v1 as components

    STATUS_COLOR = {"mapped": "#00b894", "partial": "#fdcb6e", "new": "#e17055"}
    DEFAULT_COLOR = "#74b9ff"

    net = Network(
        height="620px", width="100%",
        bgcolor="#0e1117", font_color="#fafafa",
        directed=True, notebook=False,
    )
    net.set_options("""{
        "physics": {"stabilization": {"iterations": 150},
                    "barnesHut": {"gravitationalConstant": -8000, "springLength": 180}},
        "edges":   {"font": {"size": 11, "color": "#aaaaaa"}, "smooth": {"type": "curvedCW", "roundness": 0.2}},
        "nodes":   {"font": {"size": 13}, "borderWidth": 2}
    }""")

    added: set[str] = set()
    for r in umls_results:
        term  = r["term"]
        color = STATUS_COLOR.get(r["status"], DEFAULT_COLOR)
        stys  = ", ".join(r.get("semantic_types", [])) or "unknown"
        cui   = r.get("cui") or "—"
        snomed_tip = "; ".join(f"{s['code']}" for s in r.get("snomed", []))
        tooltip = f"{term}\nStatus: {r['status']}\nCUI: {cui}\nTypes: {stys}"
        if snomed_tip:
            tooltip += f"\nSNOMED: {snomed_tip}"
        net.add_node(term, label=term, color=color, title=tooltip, size=22)
        added.add(term)

    for rel in relationships:
        src, tgt, lbl = rel.get("source",""), rel.get("target",""), rel.get("relation","")
        if not src or not tgt:
            continue
        for node in (src, tgt):
            if node not in added:
                net.add_node(node, label=node, color=DEFAULT_COLOR, size=16)
                added.add(node)
        net.add_edge(src, tgt, label=lbl, color="#555577", title=lbl)

    # Legend
    for status, color in STATUS_COLOR.items():
        net.add_node(f"[{status}]", label=status, color=color,
                     size=12, shape="box", physics=False,
                     x=-600, y=-250 + list(STATUS_COLOR).index(status) * 60)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                      mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    html = open(tmp.name, encoding="utf-8").read()
    os.unlink(tmp.name)
    components.html(html, height=640, scrolling=False)


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
    st.title("PDF Chat Assistant")
    logging.info("App started")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "doc_info_cache" not in st.session_state:
        st.session_state.doc_info_cache = {}

    # --- Sidebar: PDF library ---
    st.sidebar.markdown("### 📚 PDF Library")
    pdf_keys = client.keys("pdf:*")
    pdf_list = []
    if pdf_keys:
        for k in pdf_keys:
            data = client.get(k)
            if data:
                info = json.loads(data)
                filename = info.get("filename", "Unknown")
                key_str = k.decode() if isinstance(k, bytes) else k
                pdf_list.append({"filename": filename, "key": key_str})

    selected_filename = None
    if pdf_list:
        options = [p["filename"] for p in pdf_list]
        selected_idx = st.sidebar.radio(
            "Select PDF:",
            range(len(options)),
            format_func=lambda i: f"📄 {options[i]}",
            key="pdf_selector",
        )
        selected_pdf = pdf_list[selected_idx]
        selected_filename = selected_pdf["filename"]
        st.sidebar.caption(f"🔑 `{selected_pdf['key']}`")
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
        st.session_state["doc_info_cache"] = {}
        st.rerun()

    st.sidebar.markdown("---")
    file_upload = st.sidebar.file_uploader(
        label="Upload PDF", type=["pdf"],
        accept_multiple_files=False,
        key="pdf_uploader",
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
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
        else:
            st.sidebar.success("✅ PDF ready — ask your question!")

    # --- Main area: tabs ---
    tab_info, tab_chat = st.tabs(["📄 Document Info", "💬 Chat"])

    with tab_info:
        if selected_filename:
            st.subheader(f"📄 {selected_filename}")
            if selected_filename not in st.session_state.doc_info_cache:
                with st.spinner("Analyzing document..."):
                    try:
                        doc_info = get_document_info(selected_filename)
                        st.session_state.doc_info_cache[selected_filename] = doc_info
                    except Exception as e:
                        st.error(f"Error analyzing document: {e}")
                        doc_info = None
            else:
                doc_info = st.session_state.doc_info_cache[selected_filename]

            if doc_info:
                st.markdown("#### 📝 Summary")
                st.write(doc_info["summary"])
                st.markdown("---")
                col_toc, col_ent = st.columns(2)
                with col_toc:
                    st.markdown("#### 📋 Table of Contents")
                    st.write(doc_info["toc"])
                with col_ent:
                    st.markdown("#### 🏷️ Key Entities")
                    st.write(doc_info["entities"])

                st.markdown("---")
                st.markdown("#### 🔬 UMLS / SNOMED-CT Entity Mapping")
                cache = st.session_state.doc_info_cache[selected_filename]
                if "umls_mapping" not in cache:
                    if st.button("▶ Run UMLS Mapping", key=f"umls_btn_{selected_filename}"):
                        terms = parse_entity_terms(doc_info["entities"])
                        if not terms:
                            st.warning("No entity terms found to map.")
                        else:
                            _prog = st.progress(0, text="Starting...")
                            _status = st.empty()
                            def _cb(i, total, term):
                                _prog.progress((i + 1) / total, text=f"Mapping {i+1}/{total}: {term}")
                                _status.write(f"🔍 {term}")
                            try:
                                cache["umls_mapping"] = map_entities_to_umls(terms, progress_cb=_cb)
                            except Exception as e:
                                st.error(f"UMLS mapping error: {e}")
                            finally:
                                _prog.empty()
                                _status.empty()
                            st.rerun()
                else:
                    _render_umls_results(cache["umls_mapping"])

                    # --- Relationships section (visible only after UMLS mapping) ---
                    if "umls_mapping" in cache:
                        st.markdown("---")
                        st.markdown("#### 🔗 Entity Relationships")
                        if "relationships" not in cache:
                            if st.button("▶ Extract Relationships", key=f"rel_btn_{selected_filename}"):
                                mapped_terms = [r["term"] for r in cache["umls_mapping"]]
                                with st.spinner("Extracting relationships from document..."):
                                    try:
                                        rels = extract_relationships(mapped_terms, selected_filename)
                                        cache["relationships"] = rels
                                    except Exception as e:
                                        st.error(f"Relationship extraction error: {e}")
                                        cache["relationships"] = []
                                st.rerun()
                        else:
                            rels = cache["relationships"]
                            if rels:
                                st.caption(f"{len(rels)} relationships found — nodes colored by UMLS status: 🟢 Mapped  🟡 Partial  🔴 New")
                                render_entity_graph(cache["umls_mapping"], rels)
                            else:
                                st.info("No relationships could be extracted from the document context.")
        else:
            st.info("Upload a PDF and select it from the sidebar to view document details.")

    with tab_chat:
        if selected_filename:
            st.caption(f"Chatting about: **{selected_filename}**")

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

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