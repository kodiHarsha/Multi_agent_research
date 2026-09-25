from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from tools import web_scrape,web_search
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()

llm=ChatGroq(model="qwen/qwen3.8-27b",temperature=0.9)

# 1st agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

# 2nd agent
def Read_agent():
    return create_agent(
        model=llm,
        tools=[web_scrape]
    )

# writer chain
writer_prompt=ChatPromptTemplate.from_messages([
    
        ("system","you are an Expert writer write  a clear structured  insightful report "),
        
        ("human",
         """write a Detailed Report on topic below 
        Topic:{topic}

        Research genrator
        {research}

        structure the report as 
        1.introduction
        2.main findings 
        3.conclusion
        4.sources[list url]

        Be detailed and factual and accurate answers
        
        """
)])

writer_chain=writer_prompt|llm|StrOutputParser()

# critic chain
critic_prompt=ChatPromptTemplate.from_messages(
    [(
        "system","u are an sharp and critic research critic be honest "
    ),
    (
        "human","""review the research and evaluate it below 
         Report :{report}
          score:x/10
           strengths
            Area of improve """
    )]
)
critic_chain=critic_prompt|llm|StrOutputParser()
