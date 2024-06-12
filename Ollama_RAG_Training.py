from flask import Flask,request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
app=Flask(__name__)

MODEL="llama3"

folder_path = "db"

cached_llm=Ollama(model=MODEL)

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template("""
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an awsner from the provided information say so. [/INST] </s>  
    [INST] {input}
                            [/INST]
                                          
""")

@app.route("/ai",methods=["POST"])
def aiPOST():
    print("POST /ai called")
    json_content = request.json

    query=json_content.get("query")

    print(f"query:{query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer":response}
    return response_answer 

@app.route("/ask_pdf",methods=["POST"])
def aiPOST():
    print("POST /ask_pdf called")
    json_content = request.json

    query=json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Createing chain")
    retriever = vector_store.as_retriever(
        search_type = "sinilarity_score_threshold",
        search_kwargs = {
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm,raw_prompt)
    chain = create_retrieval_chain(retriever,document_chain)

    result= chain.invoke({"input": query})

    response_answer = {"answer" :result}
    return response_answer 

@app.route("/pdf",methods=["POST"])
def pdfPOST():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/"+file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"docs len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
        )

    response = {
        "status": "Successfully Uploaded", 
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks)
    }
    return response

def start_app():
    app.run(host="127.0.0.1" , port= 8080 , debug=True)

if __name__=="__main__":
    start_app()