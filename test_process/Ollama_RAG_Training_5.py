from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
import os

os.environ["GOOGLE_CSE_ID"] = "e57d9d4e1e9c5479e"
os.environ["GOOGLE_API_KEY"] = "AIzaSyD6AidfbIBA9vMLAAH3XafKjuMuIcDLlV0"
os.environ["TAVILY_API_KEY"] = "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"

MODEL = "llama3"
BASE_URL="http://10.2.1.36:11434/v1"

llm = ChatOpenAI(api_key="ollama",model=MODEL,base_url=BASE_URL,temperature=0)

search = TavilySearchResults(max_results=1)

tools = [search]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True,max_iterations=5,max_execution_time=1)

def start_app():
    
    while True:
        question=input("You：")
        if question == "End"  or question =="end":
            return

        response = agent_executor.invoke({"input":question})
        print(response["output"])

if __name__ == "__main__":
    start_app()
