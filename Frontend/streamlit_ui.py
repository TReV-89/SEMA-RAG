import os
import chromadb
import streamlit as st
from streamlit_folium import st_folium
import folium

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from rag_methods import (
    clear_chat,
    initialize_session_states,
    generate_response,
    display_chat_messages,
    handle_file_upload,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")


if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not in .env file.")

# To this:
# chroma_host = os.getenv("CHROMA_HOST", "rag-system-data.onrender.com")
# chroma_port = int(os.getenv("CHROMA_PORT", "443"))
# use_ssl = os.getenv("CHROMA_SSL", "true").lower() == "true"

# client = chromadb.HttpClient(host=chroma_host, port=chroma_port, ssl=use_ssl)
client = chromadb.PersistentClient(path="./chroma_db_data")

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = client.get_or_create_collection(
        name="rag_collection_user",
        metadata={
            "description": "SEMA document collection",
            "hnsw:space": "cosine",
            "hnsw:batch_size": 10000,
        },
    )

if "llm" not in st.session_state:

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.7,
        max_tokens=None,
    )
    st.session_state.llm = llm
else:
    llm = st.session_state.llm

initialize_session_states()


def get_document_info():
    try:
        collection = st.session_state.rag_collection
        # Get all metadata to extract unique sources (files)
        all_data = collection.get(include=["metadatas"])
        if all_data and all_data["metadatas"]:
            # Get unique source files
            unique_sources = set(
                metadata.get("source", "") for metadata in all_data["metadatas"]
            )
            # Extract file names from paths
            file_names = []
            for source in unique_sources:
                if "source" in source:
                    import re

                    match = re.search(r"'source':\s*'([^']*)'", source)
                    if match:
                        full_path = match.group(1)
                        file_names.append(os.path.basename(full_path))
            return len(unique_sources), file_names
        return 0, []
    except Exception as e:
        st.error(f"Error getting document info: {e}")
        return 0, []


st.title("CITIZEN FEEDBACK SYSTEM")

# Define location coordinates (Kampala area)
locations_map = {
    "Kiswa Hospital": [0.3476, 32.5825],
    "Kisugu Hospital": [0.2891, 32.6142],
    "Kampala Hospital": [0.3163, 32.5822],
    "Kira Road Police": [0.3340, 32.5950],
    "Old Kampala Police": [0.3136, 32.5675],
    "Kasangati Health Center": [0.4650, 32.6100],
    "Kabalagala Police": [0.2950, 32.5950],
    "Komamboga Health Center": [0.3850, 32.5700],
}

# Center map on Kampala
m = folium.Map(location=[0.3476, 32.5825], zoom_start=12)

# Add markers for each location
for location_name, coords in locations_map.items():
    folium.Marker(
        location=coords,
        popup=f"<b>{location_name}</b>",
        tooltip=location_name,
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

st_folium(m, width=700, height=500)


with st.expander("📍 Select a Location", expanded=True):
    loc_option = st.radio(
        "Choose your location:",
        (
            "Kiswa Hospital",
            "Kisugu Hospital",
            "Kampala Hospital",
            "Kira Road Police",
            "Old Kampala Police",
            "Kasangati Health Center",
            "Kabalagala Police",
            "Komamboga Health Center",
        ),
    )

    st.info(f"Selected: **{loc_option}**")


st.markdown("---")


doc_count, file_names = get_document_info()
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    st.metric("📚 Documents", doc_count)
with col2:
    if doc_count > 0:
        st.metric("🟢 Status", "Ready")
    else:
        st.metric("🟡 Status", "No Docs")


display_chat_messages()

if message := st.chat_input("Type your message here..."):
    if doc_count == 0:
        st.warning("⚠️ Please upload at least one document before asking questions.")
    else:
        st.session_state.messages.append(
            SystemMessage(
                content=f"The user is interacting with documents from {loc_option}."
            )
        )
        st.session_state.messages.append(HumanMessage(content=message))
        response = generate_response(message, st.session_state.rag_collection, llm)
        st.rerun()

with st.sidebar:
    st.header(" Controls")

    st.subheader(" Database Status")
    st.info(f"Documents in database: **{doc_count}**")

    # Show uploaded file names in sidebar
    if file_names:
        with st.expander("📄 Uploaded Files", expanded=False):
            for idx, file_name in enumerate(file_names, 1):
                st.write(f"{idx}. {file_name}")

    if doc_count > 0:
        st.success("✅ Ready to answer questions!")
    else:
        st.warning("⚠️ Upload documents to get started")

    if st.button("🗑️ Clear Chat", type="primary"):
        clear_chat()
        st.success("Chat cleared!")
        st.rerun()

    st.subheader("📁 Upload Documents")
    st.markdown("*Add documents to your knowledge base*")

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["xlsx"],
        help="Upload XLSX files to add to your knowledge base",
        accept_multiple_files=True,  # Enable multiple file upload
    )

    if uploaded_files:
        files_uploaded_this_run = False
        for file in uploaded_files:
            # Check if file was already processed
            if file.name not in st.session_state.processed_files:
                with st.spinner(f"Processing {file.name}..."):
                    if handle_file_upload(file, st.session_state.rag_collection):
                        st.session_state.processed_files.add(file.name)
                        st.success(f"✅ Successfully uploaded: {file.name}")
                        files_uploaded_this_run = True
                    else:
                        st.error(f"❌ Failed to upload: {file.name}")

        # Rerun to update the document count and file list
        if files_uploaded_this_run:
            st.rerun()

    # Add option to clear processed files history
    if st.session_state.processed_files:
        if st.button("Reset File Upload"):
            st.session_state.processed_files.clear()
            st.rerun()
