from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import RESOURCE_ATTRIBUTES, Resource
from typing import Dict, Any, Optional
from loguru import logger
import functools
import time

class ObservabilityManager:
    """Singleton for managing LLM tracing and observability."""
    _instance = None

    def __new__(cls, service_name: str = "agentic-os"):
        if cls._instance is None:
            cls._instance = super(ObservabilityManager, cls).__new__(cls)
            cls._instance._init_otel(service_name)
        return cls._instance

    def _init_otel(self, service_name: str):
        """Initializes OpenTelemetry tracing."""
        resource = Resource(attributes={
            "service.name": service_name
        })
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
        logger.info(f"ObservabilityManager initialized for: {service_name}")

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Starts a new span."""
        return self.tracer.start_as_current_span(name, attributes=attributes)

def trace_llm_call(func):
    """Decorator to trace LLM calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        obs = ObservabilityManager()
        span_name = f"llm_call_{func.__name__}"
        
        with obs.start_span(span_name, attributes={"function": func.__name__}) as span:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                span.set_attribute("status", "success")
                return result
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error.message", str(e))
                logger.error(f"Error in traced function {func.__name__}: {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_ms", duration * 1000)
                logger.debug(f"Span {span_name} completed in {duration:.2f}s")
                
    return wrapper

# Example usage class
class TracedAgent:
    def __init__(self, name: str):
        self.name = name
        self.obs = ObservabilityManager()

    @trace_llm_call
    def process_query(self, query: str):
        logger.info(f"Agent {self.name} processing: {query}")
        # Simulate LLM work
        time.sleep(0.5)
        return f"Response to {query}"
