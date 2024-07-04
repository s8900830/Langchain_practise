from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.utilities import GoogleSearchAPIWrapper,SearchApiAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_tool_calling_agent,create_structured_chat_agent,create_openai_tools_agent,create_openai_functions_agent
from langchain_community.agent_toolkits.load_tools import load_tools
import os
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
from fake_useragent import UserAgent

os.environ['USER_AGENT'] = UserAgent().chrome
os.environ["GOOGLE_CSE_ID"] = "e57d9d4e1e9c5479e"
os.environ["GOOGLE_API_KEY"] = "AIzaSyD6AidfbIBA9vMLAAH3XafKjuMuIcDLlV0"
os.environ["SEARCHAPI_API_KEY"] = "uAgbd1TdU5aCBoAZ2MotYWm2"
os.environ["TAVILY_API_KEY"] = "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"

MODEL = "llama3"
BASE_URL="http://10.2.1.36:11434/v1"

llm = ChatOpenAI(api_key="ollama",model=MODEL,base_url=BASE_URL,temperature=0)

search = TavilySearchResults()

tools = [search]

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

def start_app():
    
    while True:
        question=input("You：")
        if question == "End"  or question =="end":
            # for chat in chat_history:
            #     print(chat)
            return

        response = agent.invoke({"input":question})

        print("AI："+response["output"])

def test():
    print(search.invoke({"query": "What happened in the latest burning man floods"}))


if __name__ == "__main__":
    start_app()
