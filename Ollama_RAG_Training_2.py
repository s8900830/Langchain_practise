from flask import Flask, request
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.utilities import SearchApiAPIWrapper
import json
import os

app = Flask(__name__)

os.environ["SEARCHAPI_API_KEY"] = "uAgbd1TdU5aCBoAZ2MotYWm2"


MODEL = "llama3"

folder_path = "db"

cached_llm = Ollama(model=MODEL)

@app.route("/ask", methods=["POST"])
def askPOST():
    json_content = request.json

    # query = json_content.get("query")#
    query = "今天的天氣如何？"
    search = SearchApiAPIWrapper(api_key="your_api_key")
    # search_res = search.run(query)

    # print(f"query:{query}")

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="基於以下上下文回答問題：\n{context}\n問題：{question}"
    )

    # chain = cached_llm | prompt_template | search_res
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]
    chain =  initialize_agent(
        tools,
        cached_llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True)

    # 初始化會話
    session = ConversationSession(chain=chain)

    # 示例提問
    response = session.ask(query)
    return response

class ConversationSession:
    def __init__(self, chain):
        self.chain = chain
        self.history = []

    def ask(self, question):
        context = " ".join([turn["response"] for turn in self.history])
        response = self.chain.invoke({"context":context, "question":question})
        self.history.append({"question": question, "response": response})
        return response



def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
