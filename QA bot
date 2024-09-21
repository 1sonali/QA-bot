# RAG-based QA Bot Implementation in Google Colab for PDF Documents from Google Drive

# Step 1: Install required libraries
!pip install pinecone-client cohere transformers sentence-transformers PyPDF2 nltk

# Step 2: Import necessary modules
import pinecone
import cohere
from sentence_transformers import SentenceTransformer
import torch
import os
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from google.colab import drive
import glob

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Step 3: Mount Google Drive
drive.mount('/content/drive')

# Step 4: Set up API keys
os.environ['PINECONE_API_KEY'] = '5902fe79-a604-401b-95f4-299305ce9ce0'
os.environ['PINECONE_ENVIRONMENT'] = 'us-east-1-aws'
os.environ['COHERE_API_KEY'] = 'fjdr1j3FsqVGRaUBFPi4p9pmTt3lI8ffhqDkiHG2'

pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT']) # Changed from pinecone.init() to pinecone.Pinecone()
co = cohere.Client(os.environ['COHERE_API_KEY'])

# Step 6: Load and preprocess the PDF documents from Google Drive
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_into_chunks(text, chunk_size=1000, overlap=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

            # Add overlap
            overlap_text = " ".join(chunks[-1].split()[-overlap:])
            current_chunk = overlap_text + " " + current_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Specify the path to your PDF folder in Google Drive
pdf_folder_path = '/content/drive/MyDrive/pdfs'

# Process all PDF files in the specified Google Drive folder
pdf_files = glob.glob(f'{pdf_folder_path}/*.pdf')
all_chunks = []

for pdf_file in pdf_files:
    text = extract_text_from_pdf(pdf_file)
    chunks = split_into_chunks(text)
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")

# Step 7: Create embeddings and index documents
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(documents):
    return model.encode(documents)

embeddings = create_embeddings(all_chunks)

index_name = 'qa-bot-index'
if index_name not in pc.list_indexes().names(): # Use pc.list_indexes().names() to check for the index name
    index_name = 'qa-bot-index'
if index_name not in pc.list_indexes().names(): # Use pc.list_indexes().names() to check for the index name
    pc.create_index(index_name, dimension=384, spec={'serverless': {'type':'starter', 'cloud': 'aws', 'region':'us-east-1'}})  # 384 is the dimension for 'all-MiniLM-L6-v2' # Added 'pod' key to spec argument
index = pc.Index(index_name) # Use pc.Index() to get the index

# Upsert embeddings to Pinecone
for i, embedding in enumerate(embeddings):
    index.upsert([(str(i), embedding.tolist(), {"text": all_chunks[i]})])

# Step 8: Implement the RAG model
def retrieve_documents(query, top_k=3):
    query_embedding = model.encode([query])[0]
    # results = index.query(query_embedding.tolist(), top_k=top_k, include_metadata=True) # Incorrect call to index.query()
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True) # Changed to use keyword arguments
    return [result['metadata']['text'] for result in results['matches']]

def generate_answer(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = co.generate(
        model='command-nightly',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop_sequences=["Context:", "Question:"]
    )
    return response.generations[0].text.strip()

def rag_qa(query):
    relevant_docs = retrieve_documents(query)
    context = " ".join(relevant_docs)
    answer = generate_answer(query, context)
    return answer

# Step 9: Test the model
test_queries = [
    "What is the main topic of these research papers?",
    "Summarize the key findings from the papers.",
    "What are some common methodologies used in these studies?",
    "Are there any contradictory results among the papers?",
    "What future research directions are suggested by these papers?"
]

for query in test_queries:
    answer = rag_qa(query)
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    print()

# Step 10: Clean up (optional)
# Uncomment the following line if you want to delete the index after testing
# pinecone.delete_index(index_name)
