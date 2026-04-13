# app.py
import os
import warnings
warnings.filterwarnings("ignore")

import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from mcp_tools.doc_parser import _parse_pdf, _parse_docx
from rag.embedder import build_vectorstore, search_vectorstore
from graph.graph import build_graph

st.set_page_config(
    page_title="Legal Doc Analyzer",
    page_icon="⚖",
    layout="wide"
)

st.title("Legal Document Simplifier & Risk Detector")
st.caption("Upload a legal document and get a plain-English risk report instantly.")

# --- Sidebar ---
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Upload a PDF or DOCX
    2. AI extracts all clauses
    3. Each clause is risk-scored
    4. Get plain-English explanations
    5. Get questions to ask your lawyer
    """)
    st.divider()
    st.caption("Built with LangGraph + LangChain + MCP + Groq")

# --- Initialize session state ---
if "result" not in st.session_state:
    st.session_state.result = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# --- File upload ---
uploaded = st.file_uploader(
    "Upload your legal document",
    type=["pdf", "docx"],
    help="Supported formats: PDF, DOCX"
)

if uploaded:
    st.success(f"Uploaded: {uploaded.name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("File name", uploaded.name)
    col2.metric("File size", f"{round(uploaded.size / 1024, 1)} KB")
    col3.metric("File type", uploaded.name.split(".")[-1].upper())

    if st.button("Analyze Document", type="primary", use_container_width=True):
        ext = uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
            f.write(uploaded.read())
            tmp_path = f.name

        try:
            with st.status("Analyzing your document...", expanded=True) as status:
                st.write("Parsing document...")
                text = _parse_pdf(tmp_path) if ext == "pdf" else _parse_docx(tmp_path)
                st.write(f"Extracted {len(text)} characters")

                st.write("Building knowledge base...")
                st.session_state.vectorstore = build_vectorstore(text)

                st.write("Running AI analysis...")
                graph = build_graph()
                st.session_state.result = graph.invoke({
                    "raw_text": text,
                    "clauses": [],
                    "risk_report": [],
                    "simplified": [],
                    "questions": [],
                    "final_report": ""
                })
                st.session_state.analyzed = True
                status.update(label="Analysis complete!", state="complete")
        finally:
            os.unlink(tmp_path)

# --- Show results only if analysis is done ---
if st.session_state.analyzed and st.session_state.result:
    result = st.session_state.result
    vectorstore = st.session_state.vectorstore

    st.divider()

    # Risk summary
    high = sum(1 for r in result['risk_report'] if r['risk_level'] == 'high')
    medium = sum(1 for r in result['risk_report'] if r['risk_level'] == 'medium')
    low = sum(1 for r in result['risk_report'] if r['risk_level'] == 'low')

    st.subheader("Risk Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total clauses", len(result['clauses']))
    c2.metric("High risk", high, delta=f"{high} flagged", delta_color="inverse")
    c3.metric("Medium risk", medium)
    c4.metric("Low risk", low)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Risk Report")
        for risk in result['risk_report']:
            level = risk['risk_level']
            color = "🔴" if level == 'high' else "🟡" if level == 'medium' else "🟢"
            with st.expander(f"{color} Clause {risk['clause_id']} — {level.upper()}"):
                st.write(f"**Reason:** {risk['risk_reason']}")
                if risk.get('red_flags'):
                    st.write("**Red flags:**")
                    for flag in risk['red_flags']:
                        st.warning(flag)

    with right:
        st.subheader("Plain English Explanations")
        for s in result['simplified']:
            with st.expander(f"Clause {s['clause_id']} — {s['clause_type'].title()}"):
                st.info(s['simple_explanation'])

    st.divider()

    # Semantic search with LLM answer
    st.subheader("Ask a question about your document")
    query = st.text_input("e.g. What happens if I pay rent late?")
    if query and vectorstore:
        from langchain_groq import ChatGroq

        # Get relevant chunks
        results = search_vectorstore(vectorstore, query, k=2)
        context = "\n".join(results)

        # Ask LLM to answer based on context
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke(f"""You are a helpful legal assistant. 
Answer the following question in 2-3 simple sentences based only on the document context provided.
Use plain English, no legal jargon.

Document context:
{context}

Question: {query}

Answer:""")

        st.success(response.content.strip())

        with st.expander("See relevant document sections"):
            for r in results:
                st.code(r)

    st.divider()

    st.subheader("Questions to ask your lawyer")
    for i, q in enumerate(result['questions'], 1):
        st.markdown(f"**{i}.** {q}")