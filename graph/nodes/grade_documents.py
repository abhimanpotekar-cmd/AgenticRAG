from typing import Any, Dict

from graph.chains.retrival_grader import retrival_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """


    print("="*10,"Checking Documents","="*10)
    question = state["question"]
    documents = state["documents"]
    filtered_documents = []
    web_search = False
    for doc in documents:
        score = retrival_grader.invoke(
            {
                "question":question,
                "document":doc.page_content,
            }
        )
        grade=score.binary_score
        if grade.lower()=="yes":
            print("-"*10,"Grade Document Relevant","-"*10)
            filtered_documents.append(doc)
        else:
            print("-"*10,"Grade Document Not Relevant","-"*10)
            web_search = True
            continue

    return {"documents":filtered_documents,"question":question,"web_search":web_search}