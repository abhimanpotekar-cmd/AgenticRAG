from dotenv import load_dotenv
load_dotenv()
from graph.chains.retrival_grader import GradeDocuments, retrival_grader
from ingestion import retriever

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
    assert res.binary_score=="yes"

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
    assert res.binary_score=="no"