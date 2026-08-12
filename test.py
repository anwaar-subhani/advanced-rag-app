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

# Print ONLY FIRST 5 paragraph chunks with full content and details
if chunks:
    print(f"{'='*80}")
    print(f"DISPLAYING FIRST 5 PARAGRAPH CHUNKS ONLY")
    print(f"{'='*80}\n")
    
    for i, chunk in enumerate(chunks[:5]):  # Only first 5 chunks
        print(f"{'='*80}")
        print(f"PARAGRAPH CHUNK #{i+1}")
        print(f"{'-'*80}")
        print(f"Page: {chunk.metadata['page']}")
        print(f"Font Size: {chunk.metadata['font_size']}")
        print(f"Type: PARAGRAPH")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"{'-'*80}")
        print(f"Content:")
        print(f"{'-'*80}")
        print(chunk.page_content)
        print(f"{'-'*80}")
        print(f"END OF PARAGRAPH CHUNK #{i+1}")
        print(f"{'='*80}\n")
    
    # Show how many chunks were not displayed
    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more paragraph chunks not displayed")
else:
    print("No paragraph chunks found!")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"Total paragraph chunks: {len(chunks)}")
print(f"{'='*80}")