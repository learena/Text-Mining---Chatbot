
####################################################################
#                         import
####################################################################
import os
os.environ["PATH"] = os.path.join(os.path.dirname(__file__), "bin") + ":" + os.environ["PATH"]

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import glob
from pathlib import Path
import sqlite3

import chromadb.api

# Import di openai e google_genai come principali servizi LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

from langchain.schema import format_document
from dotenv import load_dotenv
load_dotenv()

#chiavi 
token1=os.getenv("OPENAI_KEY")
token2=os.getenv("GOOGLE_KEY")
token3=os.getenv("COHERE_KEY")


# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
    WebBaseLoader,
)

# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import di Chroma come vectorstore
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Cohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere

# Import streamlit
import streamlit as st

########################################################################
#              Configurazioni: servizi LLM, lingua dell'assistente, ...
########################################################################

list_LLM_providers = [
    ":rainbow[**OpenAI**]",
    "**Google Generative AI**",
]

dict_welcome_message = {
    "Italiano": "Come posso assistervi oggi?",
    "Inglese": "How can I assist you today?",
    "Francese": "Comment puis-je vous aider aujourd’hui ?",
    "Spagnolo": "¿Cómo puedo ayudarle hoy?",
    "Tedesco": "Wie kann ich Ihnen heute helfen?",
    "Russo": "Чем я могу помочь вам сегодня?",
    "Cinese": "我今天能帮你什么？",
    "Arabo": "كيف يمكنني مساعدتك اليوم؟",
    "Portoghese": "Como posso ajudá-lo hoje?",
    "Giapponese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Vectorstore backed retriever",
    "Cohere reranker",
    "Contextual compression",
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

####################################################################
#            Creazione dell'app con interfaccia streamlit
####################################################################

st.set_page_config(page_title="Conversa con il bot")
st.title("🤖 Chatbot")

# Chiavi API 
st.session_state.openai_api_key = ""
st.session_state.google_api_key = ""
st.session_state.cohere_api_key = ""


def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"],
):
    """Aggiunta di un input testuale per inserire la chiave API e di un 
    expander che contenga i possibili modelli e parametri."""
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            text_input_API_key,
            type="password",
            value=token1,
            placeholder="insert your API key",
        )
        st.session_state.google_api_key = ""

    if LLM_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key,
            type="password",
            value=token2,
            placeholder="insert your API key",
        )
        st.session_state.openai_api_key = ""

    with st.expander("**Modelli e parametri**"):
        st.session_state.selected_model = st.selectbox(
            f"Scegli modello {LLM_provider}", list_models
        )

        # parametri del modello
        st.session_state.temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )

def sidebar_and_documentChooser():
    """Creazione della sidebar e di una finestra a schede:la prima tab contiene il document chooser, mentre la seconda l'inserimento da URL."""

    with st.sidebar:
        st.caption(
            "🚀 Un chatbot basato su Retrieval Augmented Generation che utilizza 🔗 Langchain, Cohere, OpenAI e Google Generative AI"
        )
        st.write("")

        llm_chooser = st.radio(
            "Seleziona provider",
            list_LLM_providers,
            captions=[
                "[Opzioni di acquisto](https://openai.com/pricing)",
                "Limite: 60 richieste al minuto",
            ],
        )

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                LLM_provider="OpenAI",
                text_input_API_key="Inserisci chiave OpenAi - [OpenAI API Key](https://platform.openai.com/account/api-keys)",
                list_models=[
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo",
                    "gpt-4-turbo-preview",
                ],
            )

        if llm_chooser == list_LLM_providers[1]:
            expander_model_parameters(
                LLM_provider="Google",
                text_input_API_key="Inserisci chiave Google API - [Google API Key](https://makersuite.google.com/app/apikey)",
                list_models=["gemini-pro"],
            )

        # Lingua dell'assistente
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Lingua dell'assistente", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retriever")
        retrievers = list_retriever_types
        if st.session_state.selected_model == "gpt-3.5-turbo":
            retrievers = list_retriever_types[:-1]

        st.session_state.retriever_type = st.selectbox(
            f"Seleziona tipo di retriever", retrievers
        )
        st.write("")
        if st.session_state.retriever_type == list_retriever_types[1]:  # Cohere
            st.session_state.cohere_api_key = st.text_input(
                "Inserisci chiave Cohere API - [Cohere API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                value=token3,
                placeholder="Inserisci chiave Cohere API",
            )
    # Finestra a schede: Crea un nuovo Vectorstore con dei documenti| Fornisci un URL al bot
    tab_Vector_Store_With_Docs, tab_URL_bot = st.tabs(
        ["Crea un nuovo Vectorstore con dei documenti", "Fornisci un URL al bot"]
    )

    with tab_Vector_Store_With_Docs:
        # 1. Selezione documenti
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Seleziona documenti**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )
        # 2. Processing dei documenti
        st.session_state.vector_store_name = st.text_input(
            label="**I documenti saranno caricati e salvati in un vectorstore (Chroma dB). Inserisci un nome valido**",
            placeholder="Nome Vectorstore",
        )
        # 3. Pulsante per processare i documenti e creare il vectorstore
        st.button("Crea Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    with tab_URL_bot:
        st.text_input(
            "🌐 Inserisci un URL", 
            placeholder="https://example.com",
            key="rag_url",
        )
        st.button("Fornisci al bot il link", on_click=submit_url)
        

##########################################################################
#        Processing dei documenti e creazione del vectorstore (Chroma dB)
##########################################################################

def delete_temp_files():
    """Cancella tutti i file dalle cartelle './data/tmp' e './data/vector_store'"""
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    # Directory da pulire
    directories_to_clean = [TMP_DIR.as_posix(), LOCAL_VECTOR_STORE_DIR.as_posix()]
    
    for directory in directories_to_clean:
        print(f"Cleaning directory: {directory}")
        files = glob.glob(directory + "/*")  # Get all files and subdirectories
        for f in files:
            try:
                if os.path.isfile(f) or os.path.islink(f):  # If it's a file or symbolic link
                    os.remove(f)
                    print(f"Deleted file: {f}")
                elif os.path.isdir(f):  # If it's a directory
                    for root, dirs, file_names in os.walk(f, topdown=False):
                        for name in file_names:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(f)  # Remove the top-level directory
                    print(f"Deleted directory: {f}")
            except Exception as e:
                print(f"Failed to delete {f}. Reason: {e}")



def langchain_document_loader():
    """
    Creazione del caricatore di documenti per file PDF, TXT e CSV.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """

    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents


def split_documents_to_chunks(documents):
    """Divide i documenti in chunk utilizzando RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def select_embeddings_model():
    """Seleziona i modelli di embedding: OpenAIEmbeddings o GoogleGenerativeAIEmbeddings."""
    if st.session_state.LLM_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)

    if st.session_state.LLM_provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=st.session_state.google_api_key
        )
    return embeddings


def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="semilarity",
    base_retriever_k=16,
    compression_retriever_k=20,
    cohere_api_key="",
    cohere_model="rerank-multilingual-v2.0",
    cohere_top_n=10,
):
    """
    Creazione di un retriever che può essere:
        - Vectorstore backed retriever: questo è il retriever di base.
        - Contextual compression retriever: si fa il wrapping del retriever di base in un ContextualCompressionRetriever.
            Questo compressore è un Compressor Pipeline, che divide i documenti in chunk più piccoli, rimuove
            documenti ridondanti, filtra i documenti puù rilevanti, e riordina i documenti in maniera tale che i più rilevanti siano all'inzio/alla fine
            della lista.
        - Cohere_reranker: CohereRerank endpoint è utilizzato per riordinare i documenti in base alla rilevanza.

    Parametri:
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings o GoogleGenerativeAIEmbeddings.

        retriever_type (str): in [Vectorstore backed retriever,Contextual compression,Cohere reranker]. default = Cohere reranker

        base_retreiver_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retreiver_k: ritorna i vettori più simili (default k = 16).

        compression_retriever_k: i top k documenti ritornati dal compression retriever, default = 20

        cohere_api_key: Cohere API key
        cohere_model (str): modello usato da Cohere, in ["rerank-multilingual-v2.0","rerank-english-v2.0"]
        cohere_top_n: top n documenti ritornati da Cohere, default = 10

    """

    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    elif retriever_type == "Contextual compression":
        compression_retriever = create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            k=compression_retriever_k,
        )
        return compression_retriever

    elif retriever_type == "Cohere reranker":
        cohere_retriever = CohereRerank_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            cohere_model=cohere_model,
            top_n=cohere_top_n,
        )
        return cohere_retriever
    else:
        pass


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """Creazione di un vectorsore-backed retriever
    Parametri:
        search_type: Difinisce il tipo di ricerca che il retriever deve fare.
            Può essere "similarity" (default), "mmr", o "similarity_score_threshold"
        k: numero di documenti da ritornare (Default: 4)
        score_threshold: Minima soglia di rilevanza per il similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def create_compression_retriever(
    embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None
):
    """Costruzione di un ContextualCompressionRetriever.
    Possiamo fare il wrapping del base_retriever (un retriever Vectorstore-backed) in un ContextualCompressionRetriever.
    Questo compressore è un Compressor Pipeline, che divide i documenti in chunk più piccoli, rimuove
    documenti ridondanti, filtra i documenti puù rilevanti, e riordina i documenti in maniera tale che i più rilevanti siano all'inzio/alla fine
    della lista.

    Parametri:
        embeddings: OpenAIEmbeddings o GoogleGenerativeAIEmbeddings.
        base_retriever: un retriver Vectorstore-backed.
        chunk_size (int): i documenti saranno divisi in chunk usando un CharacterTextSplitter con una chunk_size di default 500.
        k (int): i top k documenti rilevanti alla domanda sono filtrati utilizzando EmbeddingsFilter. default =16.
        similarity_threshold : similarity_threshold dell' EmbeddingsFilter. default =None
    """

    # 1. splitting dei documenti in chunck più piccoli
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". "
    )

    # 2. rimozione dei documenti ridondanti
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtro basato sulla rilevanza rispetto alla domanda
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold
    )

    # 4. Riordino dei documenti

    # I documenti meno rilevanti saranno nel mezzo della lista ed i più rilevanti all'inizio/fine.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()

    # 5. Creazione di una compressor pipeline e del retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever
    )
    return compression_retriever


def CohereRerank_retriever(
    base_retriever, cohere_api_key, cohere_model="rerank-multilingual-v2.0", top_n=10
):
    """Costruzione di un ContextualCompressionRetriever utilizzando il CohereRerank endpoint per riordinare i documenti
    basati sulla rilevanza rispetto alla domanda.

    Parameters:
       base_retriever: un retriever Vectorstore-backed
       cohere_api_key: la chiave API Cohere 
       cohere_model: il modello Cohere, in ["rerank-multilingual-v2.0","rerank-english-v2.0"], default = "rerank-multilingual-v2.0"
       top_n: i top n risultati ritornati da Cohere rerank. default = 10.
    """

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model=cohere_model, top_n=top_n
    )

    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return retriever_Cohere

def submit_url():
    with st.spinner("Caricando l'URL..."):
         # Controllo degli input
        error_messages = []
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
        ):
            error_messages.append(
                f"Inserisci la tua chiave {st.session_state.LLM_provider} API"
            )

        if (
            st.session_state.retriever_type == list_retriever_types[1]
            and not st.session_state.cohere_api_key
        ):
            error_messages.append(f"Inserisci la tua chiave Cohere API")
    
        if len(error_messages) == 1:
            st.session_state.error_message = "Per favore " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Per favore "
                + ", ".join(error_messages[:-1])
                + ", e "
                + error_messages[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
        try:
            # 1. Cancellazione dei vecchi file tmp
            delete_temp_files()

            if st.session_state.rag_url is not None:
                url = st.session_state.rag_url
                documents = []

                # 2. Salvataggio dei documenti nella directory TMP_DIR
                try:
                    temp_file_path = os.path.join(TMP_DIR.as_posix(), "web_content.txt")
                    web_loader = WebBaseLoader(url)
                    documents.extend(web_loader.load())

                    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                        for doc in documents:
                            temp_file.write(doc.page_content + "\n")
                except Exception as e:
                    st.error(f"Errore durante il salvataggio dei contenuti web: {e}")

                if documents:
                    # 3. Divisione dei documenti in chunk
                    chunks = split_documents_to_chunks(documents)
                    embeddings = select_embeddings_model()

                    #  Creazione di un vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )
                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                        st.toast(f"Pagina web con URL *{url}* caricata con successo.", icon="✅")

                        #  Creazione retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )
                        #  Creazione memoria e ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                                retriever=st.session_state.retriever,
                                chain_type="stuff",
                                language=st.session_state.assistant_language,
                            )
                        #  Eliminazione della cronologia della chat
                        clear_chat_history()
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error(e)

def chain_RAG_blocks():
    """Il sistema RAG system è composto da:
    - 1. Retrieval: include i document loader, il text splitter, il vectorstore ed il retriever.
    - 2. Memoria.
    - 3. Catena di Converstaional Retrieval.
    """
    with st.spinner("Creazione vectorstore..."):
        # Check inputs
        error_messages = []
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
        ):
            error_messages.append(
                f"Inserisci la tua chiave {st.session_state.LLM_provider} API"
            )

        if (
            st.session_state.retriever_type == list_retriever_types[1]
            and not st.session_state.cohere_api_key
        ):
            error_messages.append(f"Inserisci la tua chiave Cohere API")
        if not st.session_state.uploaded_file_list:
            error_messages.append("Seleziona file da caricare")
        if st.session_state.vector_store_name == "":
            error_messages.append("Fornisci un nome per il Vectorstore")

        if len(error_messages) == 1:
            st.session_state.error_message = "Per favore " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Per favore "
                + ", ".join(error_messages[:-1])
                + ", e "
                + error_messages[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
            try:
                # 1. Cancellazione dei vecchi file tmp
                delete_temp_files()

                # 2. Upload dei documenti selezionati sulla directory temp
                if st.session_state.uploaded_file_list is not None:
                    for uploaded_file in st.session_state.uploaded_file_list:
                        error_message = ""
                        try:
                            temp_file_path = os.path.join(
                                TMP_DIR.as_posix(), uploaded_file.name
                            )
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += e
                    if error_message != "":
                        st.warning(f"Errori: {error_message}")

                    # 3. Caricamento dei documenti con loader Langchain
                    documents = langchain_document_loader()
                    print(f"Caricato/i {len(documents)} documento/i")
                    for doc in documents:
                        print(doc.metadata, doc.page_content[:100])  # Stampa dei metadati e dei primi 100 chars

                    # 4. Divisione dei documenti in chunk
                    chunks = split_documents_to_chunks(documents)
                    print(f"Creati {len(chunks)} chunks")
                    # 5. Embeddings
                    embeddings = select_embeddings_model()
                    print("Modello di embedding creato con successo")
                    test_vector = embeddings.embed_query("Test embedding")
                    print(f"Generato test embedding: {test_vector[:10]}")  # First 10 dimensions

                    # 6. Creazione di un vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )

                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                        st.info(
                            f"Vectorstore **{st.session_state.vector_store_name}** creato con successo"
                        )

                        # 7. Creazione retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )

                        # 8. Creazione memoria e ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # 9. Eliminazione della cronologia della chat
                        clear_chat_history()

                    except Exception as e:
                        st.error(e)

            except Exception as error:
                st.error(f"Si è verificato un errore: {str(error)}")


####################################################################
#                       Creazione della memoria
####################################################################


def create_memory(model_name="gpt-3.5-turbo", memory_max_token=None):
    """Creazione di un ConversationSummaryBufferMemory per gpt-3.5-turbo
    Creazione di un ConversationBufferMemory per gli altri """

    if model_name == "gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024  # max_tokens for 'gpt-3.5-turbo' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=st.session_state.openai_api_key,
                temperature=0.1,
            ),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    return memory


####################################################################
#          Creazione di ConversationalRetrievalChain con memoria
####################################################################

def answer_template(language="italian"):
    """Invio della domanda con la cronologia della chat e del contesto alla LLM che deve rispondere."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template


def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="italian",
):
    """Creazione di un ConversationalRetrievalChain.
    Per prima cosa, passa la domanda di follow-up insieme alla cronologia della chat ad una LLM che parafrasa la domanda e genera una query singola.
    Questa query è poi inviata al retriever, che prende i documenti rilevanti documents (context) e li passa insieme alla domanda singola ed alla 
    cronologia della chat ad un LLM che risponde
    """

    # 1. Definizione del prompt standalone_question.
    # Invio della domanda di follow-upstion insieme alla cronologia della chat ad una `condense_question_llm`
    # che parafrasa la domanda e genera una query singola.

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    # 2. Definizione dell' answer_prompt
    # Invio della domanda singola + la cronologia della chat + il contesto (documenti ripresi)
    # al `LLM` che risponderà

    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Aggiunta di ConversationSummaryBufferMemory per gpt-3.5, e ConversationBufferMemory per gli altri modelli
    memory = create_memory(st.session_state.selected_model)

    # 4. Instanziamento LLM: standalone_query_generation_llm & response_generation_llm
    if st.session_state.LLM_provider == "OpenAI":
        standalone_query_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
        response_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
        )
    if st.session_state.LLM_provider == "Google":
        standalone_query_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
            convert_system_message_to_human=True,
        )
        response_generation_llm = ChatGoogleGenerativeAI(
            google_api_key=st.session_state.google_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            convert_system_message_to_human=True,
        )

    # 5. Creazione del ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory


def clear_chat_history():
    """Svuota la cronologia della chat e la memoria"""
    # 1. Re-inizializzazione dei messaggi
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Eliminazione della memoria (history)
    try:
        st.session_state.memory.clear()
    except:
        pass


def get_response_from_LLM(prompt):
    """Invocazione del LLM, ottienimento di una risposta, e display dei risultati (risposta e source documents)."""
    try:
        # 1. Invocazione LLM
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        # 2. Display dei risultati
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            # 2.1. Display della risposta
            st.markdown(answer)

            # 2.2. Display dei source document:
            with st.expander("**Documenti di riferimento**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Pagina: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Fonte: "
                        + str(document.metadata["source"])
                        + page
                        + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"

                st.markdown(documents_content)

    except Exception as e:
        st.warning(e)


####################################################################
#                         Chatbot
####################################################################
def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chatta con il bot")
    with col2:
        st.button("Svuota Chat", on_click=clear_chat_history)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
        ):
            st.info(
                f"Inserisci la tua chiave {st.session_state.LLM_provider} API per continuare."
            )
            st.stop()
        with st.spinner("In esecuzione..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()
