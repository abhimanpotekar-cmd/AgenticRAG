from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.chains.retrival_grader import structured_llm_grader

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
class GradeHallucination(BaseModel):
    """Binary score for hallucination present in the generated answer"""

    binary_score:bool =Field(
        description="BAnswer is grounded in the facts 'yes' or 'no'",
    )

structured_llm_grader = llm.with_structured_output(
    GradeHallucination
)


system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human","Set of facts: \n\n {documents} \n\n  . LLM generation:{generation}")
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader