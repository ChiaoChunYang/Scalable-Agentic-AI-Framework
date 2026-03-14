import streamlit as st
import uuid
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.manager import MemoryManager
from src.core.rag_engine import RAGEngine
from loguru import logger

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine()

def main():
    st.set_page_config(page_title="Agentic OS - GTM Agent", page_icon="🤖", layout="wide")
    
    st.title("🤖 Agentic OS: GTM Enterprise Agent")
    st.markdown("---")
    
    init_session_state()
    
    # Sidebar with info and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"Session ID: {st.session_state.session_id}")
        
        st.subheader("Memory Settings")
        long_term_enabled = st.checkbox("Enable Long-term Memory", value=True)
        
        st.subheader("RAG Parameters")
        top_k = st.slider("Retrieval K", 1, 10, 5)
        hybrid_weight = st.slider("Vector vs Keyword Weight", 0.0, 1.0, 0.5)

        if st.button("Clear History"):
            st.session_state.messages = []
            st.session_state.memory_manager.short_term.clear(st.session_state.session_id)
            st.rerun()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you today?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Retrieve Context from RAG
                    context_results = st.session_state.rag_engine.query(
                        prompt, k=top_k, hybrid_weight=hybrid_weight
                    )
                    context_text = "\n".join([res.content for res in context_results])
                    
                    # 2. Get Memory Context
                    memory_ctx = st.session_state.memory_manager.get_context(
                        prompt, thread_id=st.session_state.session_id
                    )
                    
                    # 3. Formulate Response (Mocking the LLM Orchestrator call)
                    # In a real system, you would call your LLMOrchestrator here
                    response = f"I've processed your query: '{prompt}'. \n\nI found {len(context_results)} relevant documents in the knowledge base. Using these, along with your conversation history (Short-term: {len(memory_ctx['short_term'])} entries), I can confirm our GTM strategy is on track."
                    
                    # 4. Save to Memory
                    st.session_state.memory_manager.add_memory(
                        prompt, thread_id=st.session_state.session_id, long_term=False
                    )
                    st.session_state.memory_manager.add_memory(
                        response, thread_id=st.session_state.session_id, long_term=long_term_enabled
                    )
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    logger.error(f"Error in chat processing: {str(e)}")
                    st.error("An error occurred while processing your request.")

if __name__ == "__main__":
    main()
