import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Load API key
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Debug: remove this line after confirming key works
# st.write("Key loaded:", GOOGLE_API_KEY[:10] if GOOGLE_API_KEY else "EMPTY")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes, api_key):
    tmp_path = None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(file_bytes)
        tmp.close()
        tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        # Store chunks as plain text — no embeddings, no ChromaDB
        all_chunks = [doc.page_content for doc in splits]

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based only on this context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def simple_search(query, chunks, top_k=4):
            # Simple keyword search — no embeddings needed
            query_words = set(query.lower().split())
            scored = []
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                score = len(query_words & chunk_words)
                scored.append((score, chunk))
            scored.sort(reverse=True)
            return "\n\n".join(chunk for _, chunk in scored[:top_k])

        def rag_chain(inputs):
            context = simple_search(inputs["input"], all_chunks)
            return (answer_prompt | llm | StrOutputParser()).invoke({
                "input": inputs["input"],
                "chat_history": inputs["chat_history"],
                "context": context,
            })

        return rag_chain

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# --- PDF Upload ---
if not st.session_state.pdf_loaded:
    uploaded_file = st.file_uploader("Upload a PDF to get started", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF... ⏳"):
            try:
                rag_chain = process_pdf(uploaded_file.getvalue(), GOOGLE_API_KEY)
                st.session_state.rag_chain = rag_chain
                st.session_state.pdf_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# --- Chat UI ---
else:
    if st.button("📄 Upload a different PDF"):
        st.session_state.chat_history = []
        st.session_state.pdf_loaded = False
        st.session_state.rag_chain = None
        st.rerun()

    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    if question := st.chat_input("Ask a question about your PDF..."):
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_chain({
                        "input": question,
                        "chat_history": st.session_state.chat_history,
                    })
                    st.markdown(answer)
                except Exception as e:
                    answer = f"Error: {e}"
                    st.error(answer)

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))