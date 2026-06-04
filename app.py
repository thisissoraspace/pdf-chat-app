import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# --- Sidebar ---
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("Processing PDF..."):
            tmp_path = None
            if uploaded_file and not st.session_state.pdf_loaded:
                with st.spinner("Processing PDF..."):
                    tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(uploaded_file.read())
                tmp.close()
                tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)

                embedding = FastEmbedEmbeddings()
                vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
                retriever = vectorstore.as_retriever()

                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                rephrase_prompt = ChatPromptTemplate.from_messages([
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                    ("human", "Rephrase the above as a standalone question.")
                ])

                answer_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer based only on this context:\n\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                def get_context(inputs):
                    rephrased = (rephrase_prompt | llm | StrOutputParser()).invoke(inputs)
                    docs = retriever.invoke(rephrased)
                    return "\n\n".join(d.page_content for d in docs)

                def rag_chain(inputs):
                    context = get_context(inputs)
                    return (answer_prompt | llm | StrOutputParser()).invoke({
                        "input": inputs["input"],
                        "chat_history": inputs["chat_history"],
                        "context": context
                    })

                st.session_state.rag_chain = rag_chain
                st.session_state.pdf_loaded = True

            finally:                                          # ← must be at this indent
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)