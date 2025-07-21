import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_core.output_parsers import (
    JsonOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings,
)

from .prompt import (
    CITATION_SYSTEM_PROMPT,
    CONTEXTUALIZE_Q_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
)

# --- PATHS AND CONSTANTS ---
DATA_PATH = "app/data/"
CHROMA_PATH = "chroma_db"


def get_chat_model():
    """Returns the configured language model."""
    return ChatOllama(model="llama3.1:8b", temperature=0)


def create_vector_store():
    """Creates and persists a vector store from documents in the data path."""
    print("Starting to create/update vector store...")
    text_documents = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader).load()
    if not text_documents:
        print("No documents found. Vector store not updated.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs_with_index = []
    for i, doc in enumerate(text_splitter.split_documents(text_documents)):
        doc.metadata["doc_index"] = i + 1
        docs_with_index.append(doc)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    _ = Chroma.from_documents(docs_with_index, embeddings, persist_directory=CHROMA_PATH)
    print(f"Vector store updated with {len(docs_with_index)} document chunks.")


def get_retriever():
    """Loads the retriever from the persistent vector store."""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vector_store.as_retriever()


def create_rag_chain(retriever, llm):
    """Creates the complete RAG chain using LCEL."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = contextualize_q_prompt | llm | StrOutputParser() | retriever

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )
    rag_chain = qa_prompt | llm | StrOutputParser()

    citation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CITATION_SYSTEM_PROMPT),
            ("human", "Question: {input}\nAnswer: {answer}\nContext: {context}"),
        ]
    )
    citation_chain = citation_prompt | llm | JsonOutputParser()

    def format_docs(docs):
        return "\n\n".join(f"Document {doc.metadata['doc_index']}:\n{doc.page_content}" for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(context=history_aware_retriever)
        | RunnablePassthrough.assign(
            answer=(lambda x: {"context": format_docs(x["context"]), "input": x["input"]}) | rag_chain
        )
        | RunnablePassthrough.assign(
            citations=(lambda x: {"context": x["context"], "input": x["input"], "answer": x["answer"]})
            | citation_chain
        )
    )
    return rag_chain


# --- GLOBAL OBJECTS (loaded once on startup) ---
# Check if the vector store needs to be created for the first time.
if not os.path.exists(CHROMA_PATH):
    create_vector_store()

# Load the main RAG chain to be used by the API.
rag_chain = create_rag_chain(get_retriever(), get_chat_model())
