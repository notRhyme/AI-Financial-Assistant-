from crewai import Agent
from langchain.tools import tool
from .rag_system import rag_query 

MODEL_NAME = 'gpt-4o-mini' 

# --- 1. Tool Definition (STANDALONE FUNCTION) ---
@tool("Financial DB Search")
def search_financial_records(query: str) -> str:
    """
    Useful to search for banking policies, interest rates, and financial reports.
    Calls the RAG system to retrieve and summarize relevant document chunks.
    """
    # Debug print to see if tool is working
    print(f"\n[Tool Used] Searching for: {query}") 
    
    retrieved_chunks = rag_query(query)
    
    if not retrieved_chunks:
        return "No relevant information found in the financial knowledge base."
    
    context = "RETRIEVED FINANCIAL DATA:\n"
    for chunk in retrieved_chunks:
        context += f"- TEXT: {chunk.get('text', 'N/A')} | SOURCE: {chunk.get('source', 'N/A')}\n"
        
    return context

# --- 2. Agent Definitions ---
class BankingAgents:
    def planner_agent(self):
        return Agent(
            role='Senior Financial Strategist',
            goal='Analyze client questions and create a structured research plan',
            backstory="You are a veteran banking executive. You break down complex questions.",
            verbose=True,
            allow_delegation=False,
            llm=MODEL_NAME
        )

    def research_agent(self):
        return Agent(
            role='Financial Research Analyst',
            goal='Fetch accurate financial data using the Knowledge Base',
            backstory="You are a diligent data analyst. You search for specific numbers.",
            verbose=True,
            allow_delegation=False,
            # Pass the function directly
            tools=[search_financial_records], 
            llm=MODEL_NAME
        )

    def reflection_agent(self):
        return Agent(
            role='Compliance & Review Officer',
            goal='Critique the findings for accuracy, tone, and completeness',
            backstory="You are responsible for Introspection. You check for compliance.",
            verbose=True,
            allow_delegation=False,
            llm=MODEL_NAME
        )