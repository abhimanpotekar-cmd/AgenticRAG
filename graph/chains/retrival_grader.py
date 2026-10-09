#tells if the document is relevant to the question or not if no websearch flag is set true
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

class GradeDocuments(BaseModel):
    """Assigns Binary score for relevance check on retrieved documents"""

    binary_score:str =Field(
        description="Documents are relevant to the question , 'YES' or 'NO'."
    )

structured_llm_grader=llm.with_structured_output(GradeDocuments)




system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human","Retrieved document : \n\n {document} \n\n .User question:\n {question} \n")
    ]
)
retrival_grader = grade_prompt | structured_llm_grader