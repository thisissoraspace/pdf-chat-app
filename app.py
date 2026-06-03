import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = (os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", ""))

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
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(uploaded_file.read())
                tmp.close()
                tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(docs)

                embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
                retriever = vectorstore.as_retriever()

                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                rephrase_prompt = ChatPromptTemplate.from_messages([
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                    ("human", "Given the conversation above, rephrase the follow-up question to be standalone.")
                ])

                answer_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer the question based only on the context below:\n\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)
                combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)
                st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
                st.session_state.pdf_loaded = True

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        st.success(f"✅ '{uploaded_file.name}' loaded! Ask away.")

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.session_state.pdf_loaded = False
        st.session_state.rag_chain = None
        st.rerun()

# --- Chat UI ---
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    if question := st.chat_input("Ask a question about your PDF..."):
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.markdown(answer)

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))