from dotenv import load_dotenv
from typing import Any ,Dict
from graph.state import GraphState
from ingestion import retriever


load_dotenv()

def retrieve(state: GraphState,)->Dict[str,Any]:
    print("="*10,"RETRIEVE","="*10)
    question=state["question"]

    documents =retriever.invoke(question)
    return {"documents":documents,"question":question}

    # pass
