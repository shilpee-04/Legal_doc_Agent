# graph/state.py
from typing import TypedDict, List

class AgentState(TypedDict):
    raw_text: str
    clauses: List[dict]       # extracted clauses
    risk_report: List[dict]   # risk scores per clause
    simplified: List[dict]    # plain-English versions
    questions: List[str]      # suggested lawyer questions
    final_report: str