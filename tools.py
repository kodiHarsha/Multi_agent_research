
from rich import print
from langchain.tools import tool
import os
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
load_dotenv()
from tavily import TavilyClient

tavily=TavilyClient(os.getenv("TAVILY_SEARCH_API"))

@tool
def web_search(query:str)->str:
    """search the recent or current informatin and provide accurate answer Return title and url"""
    results=tavily.search(query=query,max_results=2)
    out = []

    for i in results["results"]:
        out.append(
            f"Title: {i['title']}\n"
            f"URL: {i['url']}\n"
            f"Content: {i['content']}"
        )

    return "\n\n----\n\n".join(out)

@tool
def web_scrape(url:str)->str:
    """scrape and return clear data  from the url"""
    try:
        resp=requests.get(url,timeout=5)
        soup=BeautifulSoup(resp.text,"html.parser")
        for tag in soup(["scripts","style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator=" ",strip=True)[:300]
    except Exception as e:
        return f"coudnt not scrape the url : {str(e)}"        
web_scrape.invoke("https://www.thehindu.com/news/national/ec-dissent-gyanesh-kumar-resignation-row-september-24-2026-live-updates/article71502728.ece")