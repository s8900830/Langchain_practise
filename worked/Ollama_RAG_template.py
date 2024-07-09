from langchain_community.llms import Ollama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage,HumanMessage
from langchain import hub
import os

os.environ["GOOGLE_CSE_ID"] = "e57d9d4e1e9c5479e"
os.environ["GOOGLE_API_KEY"] = "AIzaSyD6AidfbIBA9vMLAAH3XafKjuMuIcDLlV0"
os.environ["TAVILY_API_KEY"] = "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"

MODEL = "llama3"
BASE_URL="http://10.2.1.36:11434/"

llm = Ollama(model=MODEL,base_url=BASE_URL)

search = TavilySearchResults()

prompt =  hub.pull("mooncake1313/gkzxs")

tools = [search]

chat_history =[]

agent = create_react_agent(llm, tools, prompt) #创建Agent
agent_executor = AgentExecutor(agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_execution_time=10,
            max_iterations=10,
            early_stopping_method="generate")

def start_app():
    
    while True:
        question=input("You：")
        if question == "End"  or question =="end":
            return

        response = agent_executor.invoke({"question":question,"chat_history":chat_history})

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response["output"]))
        print("AI："+response["output"])

if __name__ == "__main__":
    start_app()
