import os
import sys

from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader


# ============================================================================
# 1. LOAD DOCUMENTS
# ============================================================================
pdf_files=[
    "Data/Data-Structures-in-Python.pdf",
    "Data/OOP-Workbook.pdf",
    "Data/Python Crash Course.pdf"
]

html_file=["Data/Data Types & Operators.html"]

text_files=[
    "Data/Flow-of-control-in-Python.txt",
    "Data/Python-Functions.txt"
]

All_Documents=[]
total_characters_from_all_files=0

def load_file_and_extend(file_list, loader_class, file_type):
    global All_Documents
    global total_characters_from_all_files
    
    print(f"\n ------- loading {file_type} files ---------")
    for file_path in file_list:
        
        if not os.path.exists(file_path):
            print(f"Error : {file_type} file '{file_path}' not found!!")
            
        loader=loader_class(file_path)
        documents_from_current_files=loader.load()
        All_Documents.extend(documents_from_current_files)
        
        current_file_chars=sum(len(doc.page_content) for doc in documents_from_current_files)
        total_characters_from_all_files=total_characters_from_all_files+current_file_chars
        
        print(f"loaded {len(documents_from_current_files)} documents from '{file_path}'")
        print(f"Total Character from the current file : {current_file_chars:,}")
        
load_file_and_extend(pdf_files, PyPDFLoader, "PDF")
load_file_and_extend(text_files, lambda path: TextLoader(path, encoding='cp1252'), "TXT")
load_file_and_extend(html_file, UnstructuredHTMLLoader, "HTML")

print(f"\n✓ Successfully loaded a total of {len(All_Documents)} pages/documents from all files.")
print(f"  Grand total characters across all files: {total_characters_from_all_files:,}")
            

# ============================================================================
# 2. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================================

print("============================================================================")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks=splitter.split_documents(All_Documents)

print(f"✓ Split {len(All_Documents)} documents into {len(chunks)} chunks")


# ============================================================================
#  load_dotenv()
# ============================================================================

from dotenv import load_dotenv

load_dotenv()

if os.getenv("GOOGLE_API_KEY"):
    print("✅ GOOGLE_API_KEY found")
else:
    print("❌ GOOGLE_API_KEY not found")
    print("   Create a .env file with: GOOGLE_API_KEY=your-key-here")
    
    
# ============================================================================
# 3. CREATE EMBEDDINGS
# ============================================================================   

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings=GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)
print(embeddings)

# ============================================================================
# 4. CREATE CHROMA VECTOR STORE
# ============================================================================  

from langchain_community.vectorstores import Chroma

vector_store=Chroma.from_documents(
    documents=All_Documents,
    embedding=embeddings
)
print(vector_store)

# ============================================================================
# 5. CREATE RETRIEVER
# ============================================================================

retriever=vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k':1}
)

# Query

Query="Basic Data Types & Control Flow"

results=retriever.invoke(Query)

print(f"Query: {Query}\n")

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}\n")
    
    
# MMR Retriever

mmr_retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":3,
        "fetch_k": 5,
        "lambda_mult":0.8
    }
)

Query1="what is the main difference between a list and a tuple?"
mmr_results=mmr_retriever.invoke(Query1)

print(f"Query: {Query1}\n")

for doc in mmr_results:
    print(f" - {doc.page_content}\n")
    
    

# ============================================================================
# 5. Configuring LLM
# ============================================================================

from langchain_google_genai import ChatGoogleGenerativeAI

LLM=ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    max_tokens=2000
)

test_response=LLM.invoke("Hello, How are you!")
print(test_response)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer based on the context, say that you don't know. "
    "Keep the answer concise and accurate.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

# Create the prompt template
prompt = ChatPromptTemplate.from_template(system_prompt)

# print(prompt)

# Helper function to format documents
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain using LangChain 1.0+ LCEL (LangChain Expression Language)
# This uses the pipe operator (|) to chain components together
rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve docs and format them
        "question": RunnablePassthrough()      # Pass through the question
    }
    | prompt           # Format with prompt template
    | LLM              # Generate answer with LLM
    | StrOutputParser() # Parse output to string
)

print(rag_chain)

query2 = "what is the main difference between a list and a tuple?"

print(f"Query: {query2}")
print("\nProcessing...\n")

# With LangChain 1.0+, we invoke the chain with the question directly
answer = rag_chain.invoke(query2)

print("=" * 80)
print("ANSWER:")
print("=" * 80)
print(answer)


# Wikipedia Retriever

from langchain_community.retrievers import WikipediaRetriever

wikipedia_retriever=WikipediaRetriever(
    top_k_results=2,
    doc_content_chars_max=1000
)

# Query

Query3="What is the history of Python programming language?"

results=wikipedia_retriever.invoke(Query3)

print(f"Query: {Query3}\n")

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}\n")
    print("=" * 80)
    

# ============================================================================
# 5. Hybrid Retriever (Vector store retriver + Wikipedia Retriever)
# ============================================================================

Query4="What is the history of Python programming language?"

def hybrid_retriever(query: str) -> str:
    """
    Retrieves information from both local vector store and Wikipedia.

    Args:
        query: The search query

    Returns:
        Formatted string with context from both sources
    """
    
    
    # Get results from vector store
    local_docs = retriever.invoke(Query4)

    # Get results from Wikipedia
    wiki_docs = wikipedia_retriever.invoke(Query4)

    # Combine and format
    context_parts = []

    if local_docs:
        
        # Add local docs content
        for i, doc in enumerate(local_docs, 1):
            print(f"{i}. {doc.page_content}\n")
            print("=" * 80)
        context_parts.append(doc.page_content)
        

    if wiki_docs:
        
        # Add wiki docs content
        for i, doc in enumerate(wiki_docs, 1):
            print(f"{i}. {doc.page_content}\n")
            print("=" * 80)
        context_parts.append(doc.page_content)

    return "\n\n".join(context_parts)

hybrid_retriever(Query4)
