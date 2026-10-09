from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import END, StateGraph

from pathlib import Path

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader

from graph.chains.router import RouteQuery , question_router

def decide_to_generate(state:GraphState):
    print("="*10,"Access Granted","="*10)
    if state["web_search"]:
        print("-"*10,"Decision not all docs are relevant to question","-"*10)
        return WEBSEARCH
    else:
        print("-"*10,"Decision : Generate","-"*10)
        return GENERATE

def grade_generated_answer_in_documents_and_question(state:GraphState)->str:
    print("="*10,"Check Hallucinations","="*10)
    question = state["question"]
    documents = state["documents"]
    generated = state["generation"]
    score=hallucination_grader.invoke(
        {"documents":documents,"generation":generated}

    )

    hallucination_grade = score.binary_score
    if hallucination_grade:
        print("-"*10,"Decision Generation Grounded in Documents","-"*10)
        print("-"*10,"Grade the generation vs answer","-"*10
              )
        score = answer_grader.invoke({"question":question,"generation":generated})
        answer_grade = score.binary_score
        if answer_grade:
            print("-"*10,"Decision Answer's the question","-"*10)
            return "useful"
        else:
            print("-" * 10, "Decision Answer does not address the question", "-" * 10)
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def route_question(state:GraphState)->str:
    print("="*10,"Route Question","="*10)
    question=state["question"]
    source:RouteQuery = question_router.invoke({"question":question})
    if source.datasource==WEBSEARCH:
        print("-"*10,"Route Question Web Search","-"*10)
        return WEBSEARCH
    elif source.datasource=="vectorstore":
        print("-"*10,"Route Question Vectorstore","-"*10)
        return RETRIEVE



workflow=StateGraph(GraphState)
workflow.add_node(RETRIEVE,retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(WEBSEARCH,web_search)
workflow.add_node(GENERATE, generate)

workflow.set_conditional_entry_point(route_question,{
    WEBSEARCH:WEBSEARCH,
    RETRIEVE:RETRIEVE,
})

workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS,decide_to_generate,{
    WEBSEARCH:WEBSEARCH,
    GENERATE:GENERATE
})

workflow.add_conditional_edges(
    GENERATE,grade_generated_answer_in_documents_and_question,{
        "not useful":WEBSEARCH,
        "not supported":GENERATE,
        #regenrate as the answer was not grounded in the document
        "useful":END
    }
)

workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE,END)
app=workflow.compile()



app = workflow.compile()

if __name__ == "__main__":
    png_data = app.get_graph().draw_mermaid_png()

    Path("rag_graph.png").write_bytes(png_data)

    print("Graph PNG generated successfully!")