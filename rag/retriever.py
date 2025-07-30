from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag.document_loader import pages

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)  # ✅ Correct for Document list

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)

vector_store.save_local("cbse_faiss_index")


if __name__ == "__main__":
    # View a sample of split chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n")
        print(chunk.page_content)