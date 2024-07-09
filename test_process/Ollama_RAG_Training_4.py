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
BASE_URL="http://10.2.1.36:11434/"

# search = GoogleSearchAPIWrapper()
# search = SearchApiAPIWrapper()

# tool = Tool(
#     name="google_search",
#     description="Search Google for recent results.",
#     func=search.run,
# )



# loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide", header_template={'User-Agent': UserAgent().chrome,})

# docs = loader.load()
# embeddings = OllamaEmbeddings(model=MODEL)

# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(docs)
# vector = FAISS.from_documents(documents, embeddings)
# retriever = vector.as_retriever()
# retriever_tool = create_retriever_tool(
#     retriever,
#     "langsmith_search",
#     "搜索关于LangSmith的信息。对于任何关于LangSmith的问题，你必须使用这个工具！",
# )
# llm = ChatOpenAI(api_key="ollama",model=MODEL,base_url=BASE_URL,temperature=0)
llm = Ollama(model=MODEL,base_url=BASE_URL)

# embeddings = OllamaEmbeddings(model=MODEL,base_url=BASE_URL)

search = TavilySearchResults()


# tool = Tool(
#         name="search",
#         func=search.run,
#         description="use it tool to find correct answer",
#     )


# 第三步：创建工具（检索工具）
# retriever_tool = create_retriever_tool(
#     retriever,
#     "langsmith_search",
#     "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
# )



instructions = """You are an assistant."""
# base_prompt =  hub.pull("hwchase17/react")
prompt =  hub.pull("mooncake1313/gkzxs")

tools = [search]


chat_history =[]

# base_prompt = ChatPromptTemplate.from_messages(
#     [
#     ("system", """

#         根據問題作出最佳的回答，並全程使用中文回應，你可以使用以下工具:

#         {tools}

#         根據以下格式回覆解答:

#         Question: the input question you must answer
#         Thought: you should always think about what to do
#         Action: the action to take, should be one of [{tool_names}]
#         Action Input: the input to the action
#         Observation: the result of the action
#         ... (this Thought/Action/Action Input/Observation can repeat N times)
#         Thought: I now know the final answer
#         Final Answer: the final answer to the original input question
#         Begin!
#         Question: {input}
#         Thought:{agent_scratchpad}
        
#         """),
#     ("placeholder", "{chat_history}"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
#     ]
# )
# prompt = base_prompt.partial(instructions=instructions)

from langchain_core.prompts import PromptTemplate

# prompt = PromptTemplate.from_template(
#     "你要用中文回答任何問題,您可以使用以下工具{tools},並回答你有使用什麼{tool_names},且請勿使用其它國家語系回答問題{agent_scratchpad}"
# )
# Agent 寫法有變 initialize_agent 已經是過去式，現在可以使用 create_react_agent, create_json_agent, create_structured_chat_agent 等等
# 參照 https://api.python.langchain.com/en/latest/agents/langchain.agents.initialize.initialize_agent.html

agent = create_react_agent(llm, tools, prompt) #创建Agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)

def start_app():
    
    while True:
        question=input("You：")
        if question == "End"  or question =="end":
            # for chat in chat_history:
            #     print(chat)
            return

        # response = agent_executor.invoke({"input":question,"chat_history":chat_history})
        response = agent_executor.invoke({"question":question,"chat_history":chat_history})

        # chat_history.append(HumanMessage(content=question))
        # chat_history.append(AIMessage(content=response["output"]))
        print("AI："+response["output"])

def test():
    print(search.invoke({"query": "What happened in the latest burning man floods"}))


if __name__ == "__main__":
    start_app()
