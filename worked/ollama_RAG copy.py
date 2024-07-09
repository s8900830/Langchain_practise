from langchain_community.chat_models  import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import AgentType, initialize_agent,AgentExecutor
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.agents import tool
import os,asyncio

os.environ["SEARCHAPI_API_KEY"] = "uAgbd1TdU5aCBoAZ2MotYWm2"
MODEL='llama3'

llm = ChatOllama(model=MODEL)
search = SearchApiAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)
self_ask_with_search.invoke("Who lived longer: Plato, Socrates, or Aristotle?")
