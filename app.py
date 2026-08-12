import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Load environment variables from the .env file
load_dotenv()

print("Loading PDF document and analyzing font sizes...")

# Open PDF with PyMuPDF to extract font sizes
doc = fitz.open("sports_data.pdf")

# Dictionary to store font sizes and their occurrences
font_sizes = {}
all_font_sizes = []

# Extract font sizes from pages starting from page 2 and excluding last page
for page_num in range(2, len(doc) - 1):  # Skip first 2 pages and last page
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    font_sizes[size] = font_sizes.get(size, 0) + 1
                    all_font_sizes.append(size)

# Find the lowest font size (paragraph text)
min_font_size = min(font_sizes.keys())
print(f"Font sizes found in PDF (after removing first 2 pages and last page): {sorted(font_sizes.keys())}")
print(f"Lowest font size (paragraph): {min_font_size}")

# Now extract content and chunk based on font size - ONLY store paragraph chunks
content_by_font = []
current_text = ""
current_page = 0

for page_num in range(2, len(doc) - 1):  # Skip first 2 pages and last page
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                line_text = ""
                line_fonts = []
                
                for span in line["spans"]:
                    line_text += span["text"]
                    line_fonts.append(round(span["size"], 1))
                
                if line_text.strip():
                    avg_font = round(sum(line_fonts) / len(line_fonts), 1)
                    
                    # ONLY process paragraph text (lowest font size)
                    if avg_font == min_font_size:
                        # If we have existing paragraph text, append to it
                        if current_text:
                            current_text += " " + line_text
                        else:
                            current_text = line_text
                            current_page = page_num
                    else:
                        # This is heading or other font - save the paragraph chunk if we have one
                        if current_text.strip():
                            content_by_font.append({
                                'text': current_text.strip(),
                                'font_size': min_font_size,
                                'page': current_page
                            })
                            current_text = ""

# Add the last paragraph chunk if it exists
if current_text.strip():
    content_by_font.append({
        'text': current_text.strip(),
        'font_size': min_font_size,
        'page': current_page
    })

doc.close()

# Create LangChain documents from paragraph chunks only
chunks = []
for item in content_by_font:
    chunks.append(Document(
        page_content=item['text'],
        metadata={
            'page': item['page'],
            'font_size': item['font_size'],
            'is_paragraph': True
        }
    ))

print(f"\n{'='*80}")
print(f"TOTAL PARAGRAPH CHUNKS CREATED: {len(chunks)}")
print(f"{'='*80}\n")




print("Creating vector database...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Configure the database to act as a document retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Define the hidden prompt structure for the LLM
template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Initialize the free Gemini model tier
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Helper function to stitch retrieved chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Chat with your PDF in a continuous loop
print("\n--- PDF Chatbot Initialized ---")
print("Type 'exit' or 'quit' to stop.")

while True:
    # 1. Wait for the user to type a question
    user_question = input("\nYour Question: ")

    # 2. Allow the user to break the loop and close the program
    if user_question.lower() in ['exit', 'quit']:
        print("Shutting down chatbot. Goodbye!")
        break

    # 3. Send the question through our RAG chain
    response = rag_chain.invoke(user_question)

    # 4. Clean up the output format
    if isinstance(response.content, list):
        clean_answer = response.content[0]['text']
    else:
        clean_answer = response.content

    # 5. Print the final answer to the console
    print(f"Answer: {clean_answer}")