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
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# --- PDF Upload ---
if not st.session_state.pdf_loaded:
    uploaded_file = st.file_uploader("Upload a PDF to get started", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            tmp_path = None
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            try:
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
                    ("human", "Rephrase the above as a standalone question."),
                ])

                answer_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer based only on this context:\n\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                def get_context(inputs):
                    rephrased = (rephrase_prompt | llm | StrOutputParser()).invoke(inputs)
                    retrieved_docs = retriever.invoke(rephrased)
                    return "\n\n".join(d.page_content for d in retrieved_docs)

                def rag_chain(inputs):
                    context = get_context(inputs)
                    return (answer_prompt | llm | StrOutputParser()).invoke({
                        "input": inputs["input"],
                        "chat_history": inputs["chat_history"],
                        "context": context,
                    })

                st.session_state.rag_chain = rag_chain
                st.session_state.pdf_loaded = True
                st.rerun()

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

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
                answer = st.session_state.rag_chain({
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                })
                st.markdown(answer)

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))
