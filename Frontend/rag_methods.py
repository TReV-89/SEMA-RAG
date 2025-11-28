import os
import streamlit as st
import time
import requests
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


embedding_model_name = os.getenv("EMBEDDING_MODEL")
embedding_model = SentenceTransformer(embedding_model_name)


def simple_inference(instruction, api_key, base_url="https://api.sunbird.ai"):
    """
    Send a simple inference request to Sunflower API
    """
    url = f"{base_url}/tasks/sunflower_simple"

    headers = {"Authorization": f"Bearer {api_key}"}

    data = {"instruction": instruction, "model_type": "qwen", "temperature": 0.7}

    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def clear_chat():
    if "messages" in st.session_state:
        st.session_state.messages.clear()
    if "latest_messages_sent" in st.session_state:
        st.session_state.latest_messages_sent.clear()


def initialize_session_states():
    if "latest_messages_sent" not in st.session_state:
        st.session_state.latest_messages_sent = []

    if "file_path" not in st.session_state:
        st.session_state.file_path = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def split_and_load_documents(docs, collection):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Splitting documents into chunks...")
        progress_bar.progress(20)

        chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        chunked_docs = chunker.split_text(docs[0].page_content)

        # Check if any chunks were created
        if not chunked_docs:
            st.error("No chunks created from document. Document may be empty.")
            return False

        progress_bar.progress(40)
        status_text.text("Processing document chunks...")

        texts = [doc for doc in chunked_docs]

        # Generate embeddings
        status_text.text("Generating embeddings...")
        embeddings = embedding_model.encode(texts).tolist()

        # Get existing IDs to avoid duplicates
        existing_data = collection.get()
        existing_ids = set(existing_data.get("ids", []))

        # Create unique IDs based on file name and timestamp
        import hashlib

        file_hash = hashlib.md5(str(docs[0].metadata).encode()).hexdigest()[:8]

        metadatas = [
            {
                "source": str(docs[0].metadata),
                "chunk_id": i,
            }
            for i in range(len(texts))
        ]

        # Generate unique IDs that won't conflict
        ids = [f"chunk_{file_hash}_{i}" for i in range(len(texts))]

        # Filter out any existing IDs
        new_items = [
            (text, metadata, embedding, id_)
            for text, metadata, embedding, id_ in zip(texts, metadatas, embeddings, ids)
            if id_ not in existing_ids
        ]

        if not new_items:
            st.warning("All chunks from this document already exist in the database.")
            progress_bar.empty()
            status_text.empty()
            return True

        new_texts, new_metadatas, new_embeddings, new_ids = zip(*new_items)

        progress_bar.progress(60)
        status_text.text("Adding to database...")
        collection.add(
            documents=list(new_texts),
            metadatas=list(new_metadatas),
            embeddings=list(new_embeddings),
            ids=list(new_ids),
        )

        progress_bar.progress(100)
        status_text.text("✅ Documents successfully processed and added to database!")

        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        return True

    except Exception as e:
        st.error(f"Error during document processing: {e}")
        progress_bar.empty()
        status_text.empty()
        return False


def load_document(file_path, file_name):
    try:
        if file_name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path, mode="single")
            return loader.load()
        else:
            st.error(f"Unsupported file type: {file_name}")
            return None
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None


def save_uploaded_file(file, upload_dir="SEMA_files_uploaded"):

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file_path = os.path.join(upload_dir, file.name)

    if not os.path.exists(file_path):
        file.seek(0)  # Reset file pointer to the beginning
        contents = file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

    return file_path


def generate_response(message, collection, llm):

    # Query the collection for relevant documents
    query_vector = embedding_model.encode([message]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )
    locations = [
        "kabalagala",
        "kampala",
        "kasangati",
        "kisugu",
        "kira",
        "kiswa",
        "komamboga",
        "oldkla",
    ]
    docs = []
    location_matched = []
    other_docs = []

    for metadata_list, document_list in zip(results["metadatas"], results["documents"]):
        for metadata, document in zip(metadata_list, document_list):
            matched = any(
                location.lower() in str(metadata).lower() for location in locations
            )
            if matched:
                print("Location matched in document metadata:", str(metadata))
                location_matched.append(document)
            else:
                other_docs.append(document)

    docs = location_matched + other_docs

    system_message = SystemMessage(
        content="""role: A helpful assistant that can answer the users questions given some relevant documents.
                        style_or_tone:
                        - Use clear, concise language with bullet points where appropriate.
                        instruction: 
                        -Given the some documents that should be relevant to the user's question, answer the user's question.
                        output_constraints:
                        - Only answer questions based on the provided documents.
                        - If the user's question is not related to the documents, then you SHOULD NOT answer the question. Say "The question is not answerable given the documents".
                        - Never answer a question from your own knowledge.
                        output_format:
                            - Provide answers in markdown format.
                            - Provide concise answers in bullet points when relevant."""
    )

    enhanced_system_message = f"""{system_message.content} , Use the following documents to answer the question: {"\n".join(docs)}"""

    messages = [
        SystemMessage(content=enhanced_system_message),
        HumanMessage(content=message),
    ]

    recent_messages = st.session_state.messages
    messages.extend(recent_messages)

    response = llm.invoke(messages)
    # response = simple_inference(
    #     instruction=str(messages),
    #     api_key=os.getenv("SUNBIRD_API_KEY"),
    # )
    # print(response["response"])
    st.session_state.messages.append(response)
    st.session_state.latest_messages_sent.append(response)
    return response


def display_chat_messages():
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)


def handle_file_upload(file, collection):
    try:
        # Save file
        file_path = save_uploaded_file(file)
        st.session_state.file_path = file_path

        # Load documents
        rag_documents = load_document(file_path, file.name)

        if not rag_documents:
            st.error(f"Failed to load document: {file.name}")
            return False

        # Split and load into collection - check return value
        success = split_and_load_documents(rag_documents, collection)
        if success:
            st.success(f"✅ Successfully processed: {file.name}")
            return True
        else:
            st.error(f"Failed to process chunks for: {file.name}")
            return False

    except Exception as e:
        st.error(f"Error processing file upload: {e}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return False
