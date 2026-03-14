import pytest
import asyncio
from src.agents.gtm_agent import GTMAgent
from src.core.llm_orchestrator import LLMOrchestrator, LLMConfig

@pytest.fixture
def orchestrator():
    """Provides a configured LLM orchestrator for testing."""
    config = LLMConfig(
        model_provider="anthropic",
        model_name="claude-3-5-sonnet",
        prompt_version="test-1.0"
    )
    return LLMOrchestrator(config)

@pytest.fixture
def agent(orchestrator):
    """Provides a GTM Agent instance."""
    return GTMAgent(orchestrator)

@pytest.mark.asyncio
async def test_agent_solve_returns_structure(agent):
    """
    Verifies that the agent returns the expected output structure
    after solving a task.
    """
    task = "Test market expansion to APAC."
    result = await agent.solve(task)
    
    assert "task" in result
    assert result["task"] == task
    assert "reasoning_path" in result
    assert "final_strategy" in result
    assert "metrics" in result
    assert len(result["reasoning_path"]) > 0

@pytest.mark.asyncio
async def test_agent_uses_tools(agent):
    """
    Verifies that the agent utilizes its internal tools during execution.
    """
    task = "Market sizing for EMEA."
    result = await agent.solve(task)
    
    # Check if the mock market data tool was called
    assert "data_retrieved" in result
    assert "Search results for:" in result["data_retrieved"]

@pytest.mark.asyncio
async def test_orchestrator_metrics_tracking(orchestrator):
    """
    Verifies that the orchestrator correctly tracks execution metrics.
    """
    prompt = "Simple test prompt"
    response = await orchestrator.execute(prompt)
    
    metrics = response["metrics"]
    assert metrics["latency_ms"] > 0
    assert metrics["input_tokens"] > 0
    assert metrics["status"] == "success"
