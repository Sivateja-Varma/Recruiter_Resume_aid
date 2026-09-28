from fastapi import FastAPI,UploadFile,File,HTTPException,status
import uuid
from contextlib import asynccontextmanager
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os
load_dotenv()
os.makedirs('files',exist_ok=True)

@asynccontextmanager
async def lifespan(app:FastAPI):
  app.state.llm=None
  app.state.ret={}
  app.state.root_path=None
  app.state.store={}
  yield
  app.state.llm=None
  app.state.root_path=None
  app.state.ret.clear()
  app.state.store.clear()
  

  

app=FastAPI(title="Rag Application",description="Upload files to interact with them",lifespan=lifespan)


def get_llm():
  try:
    app.state.llm=ChatGroq(model='openai/gpt-oss-120b')
    return app.state.llm
  except Exception as e:
      raise ValueError(str(e))

def get_chain(path:str):
  try:
    session_id=str(uuid.uuid4())
    loader=PyPDFLoader(path)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
    splits=splitter.split_documents(docs)

    embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.from_documents(splits,embedding)
    app.state.ret[session_id]=db.as_retriever()
    return session_id
  except Exception as e:
    raise ValueError(str(e))

           


@app.post('/documentload',summary="document loading endpoint",description="load document in pdf format, as of now app only supports pdfs 😃")
async def Loader(resume:UploadFile=File(...)):
  if resume:
    try:
        app.state.root_path=F"files/{resume.filename}"
        content=await resume.read()
        with open(app.state.root_path,'wb') as f:
            f.write(content)
        session_id=get_chain(app.state.root_path) 
        if app.state.root_path is not None:
                if os.path.exists(app.state.root_path):
                    os.remove(app.state.root_path)
                    app.state.root_path=None  
        return {"action":"document loading successfull","session_id":session_id}          
    except Exception as e:
      raise HTTPException(detail='upload again',status_code=status.HTTP_400_BAD_REQUEST)     

@app.post("/chat",summary="ask away ur questions")
def Chat(input:str,session_id:str):
  if session_id not in app.state.ret:
    raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")
  if app.state.llm is None:
    try:
       app.state.llm=get_llm()   
    except:
      raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="could't load the model")
    
  def get_Messages(session_id:str)->BaseChatMessageHistory:
        if session_id not in app.state.store:
           app.state.store[session_id]=ChatMessageHistory()
        return app.state.store[session_id]   
  
  template=ChatPromptTemplate.from_messages(
    [('system','give a ats score first, and then address any question and answer questions from the provided {context} do not hallucinate and also reconstruct and phrase everything nicely and professionally in the end and also this is very important do not produce these things ** , - , give in paras or setences , DO NOT HALLUCINATE'),MessagesPlaceholder(variable_name='chat_history'),('user','{input}')]
  )
  secret_sauce="""
  - restructure the question if it is referring to an old question in the chat repharse the question accordingly. 
  - if its a new question leave it be.
  - if u cant discern whether its a new or old question leave it be do not change. 
  """
  question_re=ChatPromptTemplate.from_messages(
        [
            ('system',secret_sauce),
            MessagesPlaceholder(variable_name='chat_history'),
            ('user','{input}')
        ]  
        ) 
  qa_chain = create_stuff_documents_chain(app.state.llm, template) 
  history_chain = create_history_aware_retriever(app.state.llm, app.state.ret[session_id], question_re)
  retrieval_chain = create_retrieval_chain(history_chain, qa_chain)
  running=RunnableWithMessageHistory(retrieval_chain,get_Messages,history_messages_key='chat_history',input_messages_key="input",output_messages_key="answer") 

  config={"configurable":{"session_id":session_id}}    
  resp=running.invoke({"input":input},config)
  edited=resp.get('answer')
  final=edited.replace("\n"," ")

  return {"response":final.replace("  "," ")}

@app.post('/clearhistory',summary="endpoint to clear the chat history",description='remove history based on session_id, document is till alive')
def Clear(session_id:str):
   if session_id in app.state.store:
      app.state.store[session_id].clear()
      return "history deleted ready to take new Messages"

@app.post('/deletesession',summary="delete ur foot print",description='delete your entire session with us document loading along with session')
def Delete(session_id:str):
   if session_id in app.state.store:
      app.state.store.pop(session_id,None)
      app.state.ret.pop(session_id,None)
      return "Session deleted thanks for using this service"
   


    
        
              

  
  


  



    
  
  

