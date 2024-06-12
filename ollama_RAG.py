from langchain_community.chat_models  import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders  import YoutubeLoader
import os,asyncio

SEARCH_KEY=os.environ["SEARCHAPI_API_KEY"] = "uAgbd1TdU5aCBoAZ2MotYWm2"

MODEL='llama3'
HOSTNAME='http://127.0.0.1:11434'

chain = ChatPromptTemplate.from_template("請用中文回答：{topic}")

async def load_youtube_videos(tool, query):
    Youtube_Search_Results = await asyncio.get_event_loop().run_in_executor(None, tool.run, query)    
    for res_url in Youtube_Search_Results:
        loader = YoutubeLoader.from_youtube_url(f"https://www.youtube.com{res_url}", add_video_info=False)
        chain = chain | loader
    
    return chain

llm = ChatOllama(model=MODEL)

async def search_query(query):
    search = SearchApiAPIWrapper()
    search_result = await asyncio.get_event_loop().run_in_executor(None, search.run, query)
    return search_result

async def main():
    query = "黃仁勳是誰?"
    search_result = await search_query(query)
    tool = YouTubeSearchTool()
    chain = await load_youtube_videos(tool, query)
    chain = chain | llm | StrOutputParser()
    print(chain.invoke({"topic":search_result}))


if __name__ == '__main__':
    asyncio.run(main())

