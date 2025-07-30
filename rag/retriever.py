# Retrieves relevant documentsfrom langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)  # ✅ Correct for Document list

# View a sample of split chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk.page_content)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.save_local("cbse_faiss_index")
