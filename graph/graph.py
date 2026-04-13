# graph/graph.py
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes import extract_clauses, analyze_risk, simplify_clauses, generate_questions

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("extract", extract_clauses)
    g.add_node("analyze", analyze_risk)
    g.add_node("simplify", simplify_clauses)
    g.add_node("questions", generate_questions)

    g.set_entry_point("extract")
    g.add_edge("extract", "analyze")
    g.add_edge("analyze", "simplify")
    g.add_edge("simplify", "questions")
    g.add_edge("questions", END)

    return g.compile()