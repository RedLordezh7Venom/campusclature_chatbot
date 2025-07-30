from langchain_community.document_loaders import PyPDFLoader
pdf_path = "cbse_courses_dataset.pdf"
# Load PDF with PyPDFLoader
loader = PyPDFLoader(pdf_path)
pages = loader.load()


if __name__ == "__main__":
    # View content of the first few pages
    for i, page in enumerate(pages[:2]):  # Adjust range if you want more pages
        print(f"\n--- Page {i+1} ---\n")
        print(page.page_content)
