from chatbot.graph import GraphConfig
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any, List
import asyncio
import streamlit as st
import uuid

async def get_chatbot_response(question: str, thread_id: str) -> str:
    """Get response from the chatbot for a given question and thread."""
    try:
        # Create graph with recursion limit configuration
        graph = GraphConfig.create_graph({
            "recursion_limit": 10  # Set the maximum depth of recursion
        })
        input_data = {"question": HumanMessage(content=question)}
        
        print(f"Sending request with thread_id: {thread_id}")
        
        result = await graph.ainvoke(
            input=input_data,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract the response based on the actual structure
        if isinstance(result, dict):
            if 'messages' in result and isinstance(result['messages'], list):
                # Find the last AIMessage in the messages list
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                        return msg.content
                    elif hasattr(msg, 'content'):
                        return msg.content
            
            # If no message found, try to return the entire result as string
            return str(result)
            
        # If result is not a dict, try to get content or convert to string
        if hasattr(result, 'content'):
            return result.content
            
        return str(result)
        
    except Exception as e:
        print(f"Error in get_chatbot_response: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

def run_chatbot():
    """Run the chatbot with a simple CLI interface."""
    graph = GraphConfig.create_graph({})
    input_data = {"question": HumanMessage(content="tell me more about Building Materials & Hardware ranges?")}
    
    result = asyncio.run(graph.ainvoke(
        input=input_data,
        config={"configurable": {"thread_id": 2}}
    ))
    
    if hasattr(result, "messages") and result.messages:
        last_message = result.messages[-1]
        if hasattr(last_message, "type") and last_message.type == "ai":
            print(f"Assistant: {last_message.content}")

def run_streamlit_chatbot():
    """Run the chatbot with a Streamlit web interface."""
    st.set_page_config(page_title="Chatbot", page_icon="💬")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    # Display chat messages
    st.title("🤖 Chatbot Assistant")
    st.write("Ask me anything about Building Materials & Hardware or other topics!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(get_chatbot_response(
                    prompt, 
                    st.session_state.thread_id
                ))
                st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run the CLI version
        asyncio.run(run_chatbot())
    else:
        # Run the Streamlit version by default
        run_streamlit_chatbot()
