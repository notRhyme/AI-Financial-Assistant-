from crewai import Agent
from crewai.tools import BaseTool 
from .rag_system import rag_query 

MODEL_NAME = 'gpt-4o-mini' 

# --- 1. Define Tool using CrewAI Native Class ---
class FinancialSearchTool(BaseTool):
    # This class definition remains the same, but now it inherits from the correct location.
    name: str = "Financial DB Search"
    description: str = (
        "Useful to search for banking policies, interest rates, and financial reports. "
        "Accepts a query string and returns relevant document chunks."
    )

    def _run(self, query: str) -> str:
        # Debug print to confirm the tool is actually being called
        print(f"\n[Tool Activity] Searching Knowledge Base for: '{query}'")
        
        # Call the RAG system
        retrieved_chunks = rag_query(query)
        
        if not retrieved_chunks:
            return "No relevant information found in the financial knowledge base."
        
        # Format the output for the Agent
        context = "RETRIEVED FINANCIAL DATA:\n"
        for chunk in retrieved_chunks:
            # Using .get for safe access
            text = chunk.get('text', 'N/A')
            source = chunk.get('source', 'N/A')
            context += f"- TEXT: {text} | SOURCE: {source}\n"
            
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
        # Instantiate the tool
        search_tool = FinancialSearchTool()
        
        return Agent(
            role='Financial Research Analyst',
            goal='Fetch accurate financial data using the Knowledge Base',
            backstory="You are a diligent data analyst. You search for specific numbers.",
            verbose=True,
            allow_delegation=False,
            # Pass the instantiated tool
            tools=[search_tool], 
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
