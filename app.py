import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set page title
st.title("PDF Upload and FAISS Implementation with Snippet Matching")

# Extracted data storage
file_chunks = []
file_chunk_names = []

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into smaller chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"Uploaded {uploaded_file.name} successfully.")
        
        # Extract text from the uploaded file
        file_path = f"temp_{uploaded_file.name}"
        text = extract_text_from_pdf(file_path)
        st.write(f"Extracted text from {uploaded_file.name}:")
        
        # Split text into chunks and store the chunks
        chunks = split_text_into_chunks(text, chunk_size=500)
        for i, chunk in enumerate(chunks):
            file_chunks.append(chunk)
            file_chunk_names.append(f"{uploaded_file.name} - Chunk {i+1}")

# Set up the Sentence Transformer model
model_name = "all-MiniLM-L6-v2"  # You can adjust this to a different model
embeddings_model = SentenceTransformer(model_name)

# Convert chunks into embeddings if there are any file chunks
if file_chunks:
    chunk_vectors = embeddings_model.encode(file_chunks).astype('float32')

    # Create a FAISS index using the chunk vectors
    dimension = chunk_vectors.shape[1]  # Embedding dimension
    faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean distance)
    faiss_index.add(chunk_vectors)  # Add chunk embeddings to the index

# User query input
st.write("Ask a question based on the uploaded PDFs:")
user_query = st.text_input("Enter your question here")

# Search and display results
if user_query and file_chunks:
    # Generate embedding for user query
    query_vector = embeddings_model.encode(user_query).astype('float32')

    # Perform FAISS search
    D, I = faiss_index.search(np.array([query_vector]), k=5)  # Get top 5 matches

    # Track displayed file names to avoid duplicates
    displayed_files = set()

    # Display results with matched text snippets
    for idx in I[0]:
        if idx < len(file_chunk_names) and file_chunk_names[idx] not in displayed_files:
            # Avoid duplicate displays
            displayed_files.add(file_chunk_names[idx])

            # Display the matched file and chunk
            st.write(f"Matched File: {file_chunk_names[idx]}")

            # Display a snippet of the matching text
            snippet = file_chunks[idx]
            
            # Optionally highlight the user query in the matched snippet
            highlighted_snippet = snippet.replace(user_query, f"**{user_query}**")
            st.write(f"Matching Text Snippet: {highlighted_snippet[:500]}")  # Display first 500 characters
