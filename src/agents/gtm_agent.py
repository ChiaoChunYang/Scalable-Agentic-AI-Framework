from typing import List, Dict, Any, Optional
from loguru import logger
from src.core.llm_orchestrator import LLMOrchestrator, LLMConfig
from src.memory.manager import MemoryManager
from src.core.rag_engine import RAGEngine
from infrastructure.observability import trace_llm_call

class Tool:
    """Represents a capability the agent can use."""
    def __init__(self, name: str, description: str, func: Any):
        self.name = name
        self.description = description
        self.func = func

    def call(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class GTMAgent:
    """
    Go-To-Market AI Agent designed for strategic planning.
    Features: Multi-step reasoning (ReAct), tool-use, and persona-driven outputs.
    """
    
    def __init__(self, orchestrator: LLMOrchestrator, memory_manager: Optional[MemoryManager] = None, rag_engine: Optional[RAGEngine] = None):
        self.orchestrator = orchestrator
        self.memory_manager = memory_manager or MemoryManager()
        self.rag_engine = rag_engine or RAGEngine()
        self.tools = self._initialize_tools()
        logger.info("GTM Agent initialized with strategic planning, memory, and RAG capabilities.")

    def _initialize_tools(self) -> List[Tool]:
        """Sets up the agent's toolbelt."""
        return [
            Tool(
                name="MarketSearch",
                description="Searches for market trends and competitor analysis.",
                func=lambda q: f"Search results for: {q}. Found 3 main competitors: X, Y, Z."
            ),
            Tool(
                name="AnalyticsEngine",
                description="Runs revenue projections and TAM/SAM/SOM calculations.",
                func=lambda m: f"Projected revenue for market {m}: $50M/yr."
            )
        ]

    @trace_llm_call
    async def solve(self, task: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Solves a GTM task using multi-step reasoning, RAG context, and memory.
        """
        logger.info(f"Solving GTM task: {task} in thread {thread_id}")
        
        # 1. Retrieve Context from RAG
        context_results = self.rag_engine.query(task, k=3)
        context_text = "\n".join([res.content for res in context_results])
        
        # 2. Get Conversation History from Memory
        memory_ctx = self.memory_manager.get_context(task, thread_id=thread_id)
        history_text = "\n".join([m.content for m in memory_ctx["short_term"]])
        
        # 3. Multi-step reasoning
        reasoning_step_1 = f"Thought: I'll use the retrieved context and history to analyze {task}."
        
        # 4. Tool execution (Simulated)
        market_data = self.tools[0].call(f"Market analysis for {task}")
        
        # 5. Synthesis with LLM Orchestrator
        prompt = (
            f"Context from KB: {context_text}\n"
            f"History: {history_text}\n"
            f"Market Data: {market_data}\n"
            f"Task: {task}\n"
            "Generate a comprehensive GTM strategy."
        )
        
        llm_response = await self.orchestrator.execute(prompt)
        
        # 6. Persist to Memory
        self.memory_manager.add_memory(task, thread_id=thread_id)
        self.memory_manager.add_memory(llm_response["content"], thread_id=thread_id, long_term=True)
        
        return {
            "task": task,
            "reasoning_path": [reasoning_step_1],
            "data_retrieved": market_data,
            "final_strategy": llm_response["content"],
            "metrics": llm_response["metrics"]
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = LLMConfig(model_provider="anthropic", model_name="claude-3-5-sonnet")
        orchestrator = LLMOrchestrator(config)
        agent = GTMAgent(orchestrator)
        
        result = await agent.solve("Launch a new AI-powered observability tool in EMEA.")
        print(result)

    asyncio.run(main())
