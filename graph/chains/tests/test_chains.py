from dotenv import load_dotenv
load_dotenv()
from graph.chains.retrival_grader import GradeDocuments, retrival_grader
from ingestion import retriever
from pprint import pprint
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination


def test_retrival_grader_ans_yes() ->None:
    # pass

    question ="agent memory"
    docs=retriever.invoke(question)
    doc_content = docs[0].page_content
    res:GradeDocuments = retrival_grader.invoke(
        {
            "question":question,
            "document":doc_content,

            }
    )
    #cause for some eason the modell returns yes no even after specified to return YES NO
    assert res.binary_score.lower()=="yes"

def test_retrieval_grader_ans_no() ->None:
    # pass
    question="agent memory"
    docs=retriever.invoke(question)
    doc_content = docs[0].page_content
    res:GradeDocuments = retrival_grader.invoke(
        {
            "question":"What is better to watch AOT or ONEPIECE",
            "document":doc_content,
        }
    )
    assert res.binary_score.lower()=="no"


def test_generation_chain()->None:
    question="agent memory"
    docs = retriever.invoke(question)
    generation =generation_chain.invoke({"context":docs,"question":question})
    pprint(generation)


def test_hallucination_grader_ans_yes()->None:
    question="agent memory"
    docs = retriever.invoke(question)
    generation =generation_chain.invoke({"context":docs,"question":question})
    res:GradeHallucination = hallucination_grader.invoke(
        {
            "documents":docs,
            "generation":generation,


        }
    )

    assert res.binary_sccore

def test_hallucination_grader_ans_no() -> None:
    question = "agent memory"

    docs = retriever.invoke(question)

    # Deliberately fabricated answer
    generation = (
        "AI agents store all their memories inside "
        "physical chocolate doughnuts."
    )

    res: GradeHallucination = hallucination_grader.invoke({
        "documents": docs,
        "generation": generation
    })

    print("Hallucination score:", res.binary_sccore)

    assert res.binary_sccore is False