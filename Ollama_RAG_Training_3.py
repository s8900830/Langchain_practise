from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

llm = Ollama(model="llama3",base_url='http://10.2.1.36:11434')

chat_history =[]

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一個輔助回答機器人，你要用中文回答一切問題 並給予使用者任何意見"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}"),
    ]
)

chain = prompt_template | llm 

def start_app():
    while True:
        question=input("You：")
        if question == "End"  or question =="end":
            for chat in chat_history:
                print(chat)
            return
        # response = llm.invoke(question)
        response = chain.invoke({"input":question,"chat_history":chat_history})
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))
        print("AI："+response)

if __name__ == "__main__":
    start_app()
