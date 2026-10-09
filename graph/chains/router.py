from dotenv import load_dotenv
load_dotenv()
from typing import Literal
#with Literal a var can take one of predefined values
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
class RouteQuery(BaseModel):

    datasource:Literal["vectorstore","websearch"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or websearch",
    )


system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

structured_llm_router = llm.with_structured_output(RouteQuery)

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human","{question}")
    ]
)


question_router = route_prompt | structured_llm_router