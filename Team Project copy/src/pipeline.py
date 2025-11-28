from crewai import Crew, Process, Task
from .agents import BankingAgents

# --- 1. Task Definitions (Replaces src/tasks.py) ---
class BankingTasks:
    def plan_task(self, agent, user_question):
        return Task(
            description=f"""
            Analyze the following user question: "{user_question}".
            Identify the key financial terms and break this down into a step-by-step research plan.
            """,
            expected_output="A bullet-point list of search queries and research steps.",
            agent=agent
        )

    def research_task(self, agent, context_from_plan):
        return Task(
            description="""
            Using the research plan, use the 'Financial DB Search' tool to gather the necessary information.
            Summarize the findings into a clear, factual draft.
            """,
            expected_output="A summary of financial facts and data retrieved from the database.",
            agent=agent,
            context=[context_from_plan]
        )

    def review_task(self, agent, context_from_research):
        return Task(
            description="""
            REVIEW the draft provided by the Research Analyst. Ensure the tone is professional (Banking standard),
            and verify that the answer directly addresses the original user goal. Synthesize the final answer.
            """,
            expected_output="A final, polished, professional response to the user's question.",
            agent=agent,
            context=[context_from_research]
        )

# --- 2. Pipeline Implementation (run_pipeline) ---
def run_pipeline(user_question):
    """
    This function orchestrates the Planner -> Researcher -> Reflector flow.
    """
    
    # Instantiate Agent and Task definitions
    agents = BankingAgents()
    tasks = BankingTasks()

    # Initialize Agents
    planner = agents.planner_agent()
    researcher = agents.research_agent()
    reflector = agents.reflection_agent()

    # Initialize Tasks and link them sequentially using context
    plan_task = tasks.plan_task(planner, user_question)
    research_task = tasks.research_task(researcher, plan_task)
    review_task = tasks.review_task(reflector, research_task)

    # Define the Crew
    banking_crew = Crew(
        agents=[planner, researcher, reflector],
        tasks=[plan_task, research_task, review_task],
        process=Process.sequential,
        verbose=True
    )

    # Kickoff the process
    result = banking_crew.kickoff()
    
    return result

# this main block is for local testing feel free to remove it if necessary
if __name__ == "__main__":
    print("--- Testing run_pipeline directly ---")
    question = "What are the latest compliance requirements for new credit products?"
    final_output = run_pipeline(question)
    print("\n\nFINAL OUTPUT:\n", final_output)