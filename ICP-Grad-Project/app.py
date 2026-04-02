#Importing libraries
from pathlib import Path #(for handling paths)
from dotenv import load_dotenv #(For API Key)
import os
from langchain_community.document_loaders import TextLoader #(Reads .txt files and converts them into Langchain document objects)
from langchain_text_splitters import RecursiveCharacterTextSplitter #(Splits texts (chunks) into smaller pieces for better performance)
from langchain_core.vectorstores import InMemoryVectorStore #(A Vector DB, stored in RAM --> CHROMA, FAISS, WEAVIATE)
#from langchain_ollama import ChatOllama, OllamaEmbeddings #(ChatOllam connects LangChain to local LLM running on Ollama like qwen, OllamaEmbeddings connects LangChai to an embedding model (text -> numbers))
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import pyttsx3

#Loading the API Keys
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#Locating the Documents
dataPath = Path("Data")
docs = []

#Loading and Storing the Documents
for  filePath in dataPath.glob("*.txt"): #(Retrieve every file ending with .txt)
    text_loader = TextLoader(str(filePath), encoding = "utf-8") #A textloader obj, "utf-8" allows reading of multilingual txt
    docs.extend(text_loader.load()) #Loading and Inserting the file in the docs
    print("Loaded ", len(docs), " document's")

#Chunking

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 150)
#RecursiveCharacterTextSplitter (maintains context while splitting)
#chunk_size = 800 (each text is 800 char. long)
#chunk_overlap = 150 (Overlapping between chunks)

chunks = text_splitter.split_documents(docs)
#Chunking is performed on the docs and stored in splits
print("Created ", len(chunks), " chunks.")

#Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
#(Converts text chunks into vectors(numeric representation))

vector_store = InMemoryVectorStore(embeddings) #Storage of vectors in memory
vector_store.add_documents(chunks)

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0)
#temperature = randomness, 0 means deterministic and 1 means creative, Rag means low temp

print("\nRAG assistant is ready. Type 'exit' to quit.\n")

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

chat_history = []
while True:
    qn = input("You: ").strip()
    if qn.lower() == "exit":
        break
    retrieved_docs = vector_store.similarity_search(qn, k = 3)
    #Returns top 3 similar chunks

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')} \nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    history_text = "\n".join(chat_history)

#     prompt = f""" 
# You are an ICP Service assistant. You are required to be professional but also polite and friendly. 
# You are supposed to analyze the prompt and also the retrieved text properly before answering. Understand And Answer, do not memorize.
# Answer ONLY using the retrieved context below, by understanding and answering in your own easy words.
# Answer in English.
# If the answer is not in the context, say:
# "I don't know based on the provided documents."
# Retrieved context:
# {context}
# User question:
# {qn}"""

# Build the full prompt

    prompt = f"""
You are Hala, the official AI assistant for the Federal Authority for Identity, Citizenship, Customs & Port Security (ICP) in the UAE.
Your are to be located at the Federal Authority For Identity And Citizenship Center in Al Jarf Region - Al Jamia Street, Opposite Ajman Naturalization and Residency - Ajman.
The center's working hours is 7 AM to 5 PM from Monday till Thursday, and 7AM to 12 PM, 2:30 PM to 6 PM on Friday.
Users can contact the center at (600) 522222"
Customers from various will come to you in the center, where you will be present and answering any ICP related Queries.
You can also visit the website: https://icp.gov.ae/en/ for further info. 
The customers that require your help are quite sensitive, so please reply in a more polite and soft tone. 
Your role is to help customers at ICP service centers with:
- New ID card issuance (NEW_ID_ISSUANCE)
- ID card renewal (ID_RENEWAL)
- Lost or damaged ID card replacement (LOST_DAMAGED_ID)
- Exemption from delay fees (DELAY_FEE_EXEMPTION)
- Fee refunds (FEE_REFUND)
- Updating personal data (DATA_UPDATE)

FEES (always quote these exactly):
- UAE Citizen: AED 100 for 5-year ID, AED 200 for 10-year ID
- GCC National: AED 100 for 5-year ID
- UAE Resident: AED 100 per year of residence permit
- Lost/Damaged replacement: AED 300
- Urgent service: AED 150 extra
- Late renewal penalty: AED 20 per day, maximum AED 1000

DOCUMENTS NEEDED (New ID / Renewal):
- All customers: Personal photo (4.5x3.5cm white background), passport copy
- Residents: Birth certificate (under 15), valid residence permit
- GCC Nationals: Valid Gulf ID card

BEHAVIOR RULES:
1. Always be polite, clear, and professional.
2. Keep responses concise — customers are at a service center.
3. Ask which service the customer needs if not specified.
5. Guide the customer step by step.
6. If the user asks questions completely unrelated to the ICP services or the center then reply with I cannot help you with this in a friendly way. 
7. Only discuss ICP identity and residency services. 

Conversation so far:
{history_text}

Retrieved context:
{context}

Visitor message:
{qn}"""
    
    response = llm.invoke([{"role": "user", "content": prompt}])
    chat_history.append(f"User: {qn}")
    chat_history.append(f"Hala (ICP): {response}")
    print("Hala (ICP Assistant):", response.content, "\n")
    speak(response.content)
