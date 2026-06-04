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

        # Larger chunks = fewer embeddings = less memory
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
        splits = splitter.split_documents(docs)

        embedding = FastEmbedEmbeddings()

        # Use in-memory vectorstore to avoid disk issues
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name="pdf_chat",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

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

        return rag_chain

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# --- PDF Upload ---
if not st.session_state.pdf_loaded:
    uploaded_file = st.file_uploader("Upload a PDF to get started", type="pdf")

    if uploaded_file:
        with st.spinner("Processing PDF... this may take a moment ⏳"):
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
