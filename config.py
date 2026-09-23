from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src import config
from src.vector_store import get_retriever

llm = ChatOpenAI(
    model=config.CHAT_MODEL,
    temperature=0,
    api_key=config.OPENAI_API_KEY,
)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, just say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)


# this is the "state" that gets passed between nodes in the graph
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str


def retrieve_node(state: RAGState) -> RAGState:
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def generate_node(state: RAGState) -> RAGState:
    context_text = "\n\n".join(doc.page_content for doc in state["documents"])
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": state["question"]})
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def ask_question(question: str) -> str:
    graph = build_graph()
    result = graph.invoke({"question": question})
    return result["answer"]
