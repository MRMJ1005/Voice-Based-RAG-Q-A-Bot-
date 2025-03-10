# Importing Dependencies 
import os
import whisper
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document 
from langchain.schema.runnable import RunnableLambda
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import streamlit as st 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub

load_dotenv()

# API keys related 

# API keys have been removed due to safety issues



llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

# Details related to voice assistant 
AUDIO_FILE = "user_input.wav"
RESPONSE_AUDIO_FILE = "response.wav"  
PDF_FILE = "Insurance_Handbook_20103.pdf"  
SAMPLE_RATE = 16000
WAKE_WORD = "Hi"  
SIMILARITY_THRESHOLD = 0.4  
MAX_ATTEMPTS = 5 

# For recording audio input

def record_audio(filename, duration=10, samplerate=SAMPLE_RATE,verbose=True):
    st.write("Listening... Speak now!")
    audio =sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  
    st.write("Recording finished.")
    write(filename, samplerate, (audio * 32767).astype(np.int16))
# Transcribe the Input audio into text 
def transcribe_audio(filename):
    st.write("Transcribing audio...")
    model = whisper.load_model("base.en")
    result = model.transcribe(filename)
    return result["text"].strip().lower()
def pdf_to_documents(pdf_file):
    """Extracts text from a PDF and converts it into LangChain Document format."""
    # pdf_reader = PdfReader(pdf_file)
    documents = []

    for i, page in enumerate(pdf_file.pages):
        try:
            
            text = page.extract_text()
            if text:
                doc = Document(
                    page_content=text,  # The actual text of the PDF page
                    # metadata={"source": pdf_file.name, "page": i + 1}  # Metadata for tracking
                )
                documents.append(doc)
        except MemoryError:
            
            st.error("The uploaded PDF is too large to process. Try uploading a smaller file.")
            return []
                
        return documents 

system_prompt=("You are an assitant for question - answering tasks.Answer the questions by using the {context}.")
prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("user","{input}")
])

# Front End PART 

st.title("RAG DOCUMENT Q&A USING VOICE ASSISTANCE 📁⏺️🔊")
st.write("Hi!! This is a Voice assisted RAG ChatBot 🤖. Upload a PDF file to start the process😊")

uploaded_file=st.file_uploader("Choose a file", type =["pdf"])
rag_chain = RunnableLambda(lambda x: None)
output_text=""
final_output=""
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
if uploaded_file is not None and st.session_state.retriever is None :
    st.success("File uploaded successfully!")
    
    pdf_reader = PdfReader(uploaded_file)
    
    pdf_to_doc=pdf_to_documents(pdf_reader)
    
    final_documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(pdf_to_doc)
    
    vector_db =FAISS.from_documents(final_documents,embedding=embeddings)
    
    st.session_state.retriever  =vector_db.as_retriever()
    
    
    st.success("Retriever is ready")
    st.write("Record your question")
    
if st.session_state.retriever:
    
    if st.button("Record 🎙️"):  
        # st.write("Recording started!")
        record_audio("output.wav",duration=10)
        # st.write("Recording DONE!! Transcribing the audio ! Please wait ")
        # st.write(transcribe_audio("output.wav"))
        output_text=transcribe_audio("output.wav")
        st.write(output_text)
        question_answer_chain =create_stuff_documents_chain(llm,prompt)

        rag_chain =create_retrieval_chain(st.session_state.retriever,question_answer_chain)

        final_output=rag_chain.invoke({"input":output_text})
        st.write(final_output['answer'])
        
