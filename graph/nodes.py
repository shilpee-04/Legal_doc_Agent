# graph/nodes.py
import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from graph.state import AgentState

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def extract_clauses(state: AgentState) -> AgentState:
    print("Extracting clauses...")
    response = llm.invoke(f"""Extract all legal clauses from this document.
Return ONLY a JSON list, no explanation, no markdown. Each item must have:
- clause_id (int)
- clause_type (e.g. termination, liability, payment, non-disclosure)
- clause_text (the original text)

Document:
{state['raw_text']}""")

    try:
        json_str = re.search(r'\[.*\]', response.content, re.DOTALL).group()
        state['clauses'] = json.loads(json_str)
        print(f"Found {len(state['clauses'])} clauses")
    except Exception as e:
        print(f"Error extracting clauses: {e}")
        state['clauses'] = []
    return state

def analyze_risk(state: AgentState) -> AgentState:
    print("Analyzing risk...")
    clauses_text = "\n".join(
        f"Clause {c['clause_id']} ({c['clause_type']}): {c['clause_text']}"
        for c in state['clauses']
    )
    response = llm.invoke(f"""Analyze each clause for legal risk.
Return ONLY a JSON list, no explanation, no markdown. Each item must have:
- clause_id (int)
- risk_level (exactly one of: low, medium, high)
- risk_reason (one sentence explaining why)
- red_flags (list of strings, can be empty)

Clauses:
{clauses_text}""")

    try:
        json_str = re.search(r'\[.*\]', response.content, re.DOTALL).group()
        state['risk_report'] = json.loads(json_str)
        print(f"Risk analysis done for {len(state['risk_report'])} clauses")
    except Exception as e:
        print(f"Error analyzing risk: {e}")
        state['risk_report'] = []
    return state

def simplify_clauses(state: AgentState) -> AgentState:
    print("Simplifying clauses...")
    simplified = []
    for clause in state['clauses']:
        response = llm.invoke(
            f"Explain this legal clause in simple plain English in 2-3 sentences. "
            f"Avoid legal jargon. Write as if explaining to a friend:\n{clause['clause_text']}"
        )
        simplified.append({
            "clause_id": clause['clause_id'],
            "clause_type": clause['clause_type'],
            "simple_explanation": response.content.strip()
        })
    state['simplified'] = simplified
    print(f"Simplified {len(simplified)} clauses")
    return state

def generate_questions(state: AgentState) -> AgentState:
    print("Generating lawyer questions...")
    high_risk = [r for r in state['risk_report'] if r['risk_level'] == 'high']
    medium_risk = [r for r in state['risk_report'] if r['risk_level'] == 'medium']
    concerning = high_risk + medium_risk

    if not concerning:
        state['questions'] = ["This document appears low risk. Ask a lawyer to confirm before signing."]
        return state

    response = llm.invoke(f"""Based on these risky clauses, generate 5 specific questions 
a non-lawyer should ask before signing this document.
Return ONLY a JSON list of strings, no explanation, no markdown.

Risky clauses:
{json.dumps(concerning, indent=2)}""")

    try:
        json_str = re.search(r'\[.*\]', response.content, re.DOTALL).group()
        state['questions'] = json.loads(json_str)
        print(f"Generated {len(state['questions'])} questions")
    except Exception as e:
        print(f"Error generating questions: {e}")
        state['questions'] = []
    return state