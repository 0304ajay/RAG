# PDF Upload and FAISS-based Text Matching

This application allows users to upload PDF files, extract their content, and perform a similarity search using FAISS (Facebook AI Similarity Search). The application uses the `SentenceTransformer` model to create embeddings of text chunks from the PDFs and returns the most relevant text snippets in response to user queries. It is implemented using Streamlit for the user interface.

## Features

- **PDF Upload**: Users can upload one or more PDF files for processing.
- **Text Extraction**: The text content of each uploaded PDF is extracted.
- **Text Chunking**: The extracted text is split into smaller chunks to facilitate efficient text search.
- **Sentence Embedding**: Each chunk of text is transformed into an embedding vector using the `SentenceTransformer` model (`all-MiniLM-L6-v2`).
- **FAISS Indexing**: The embedding vectors are indexed using FAISS for fast similarity search.
- **Query Matching**: Users can input a query, and the application will return the most relevant chunks of text from the uploaded PDFs, along with the file name and a snippet of the matching content.
- **Snippet Highlighting**: The application highlights the user query in the matching text snippet for better readability.

## How It Works

1. **PDF Upload**: Users can upload one or multiple PDF files using the file uploader component.
2. **Text Extraction**: The application reads the uploaded PDFs, extracts the text from each page, and combines it into a single string.
3. **Text Chunking**: The text is split into smaller chunks (500 words by default) to ensure the embeddings are manageable and improve search efficiency.
4. **Embedding Generation**: Each chunk is converted into a numerical vector (embedding) using the `SentenceTransformer` model.
5. **FAISS Index Creation**: A FAISS index is created using the chunk embeddings, which allows for efficient similarity searches.
6. **User Query**: The user can input a search query. The query is also converted into an embedding and compared against the indexed chunks.
7. **Result Display**: The top 5 most relevant text chunks are displayed, along with their corresponding file names and highlighted text snippets where the query matches.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-faiss-matcher.git

## Running
1. streamlit run app.py
2. Open the Link 