from chatbot.graph import GraphConfig
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any, List
import asyncio
import streamlit as st
import uuid

# Create a singleton graph instance at module level
_graph_config = GraphConfig()
_graph = _graph_config.create_graph({
    "recursion_limit": 10  # Set the maximum depth of recursion
})

async def get_chatbot_response(question: str, thread_id: str, conversation_history=None) -> str:
    """Get response from the chatbot for a given question and thread."""
    try:
        # Use the singleton graph instance
        input_data = {"question": HumanMessage(content=question)}
        
        # If conversation history is provided, use it
        if conversation_history:
            input_data["conversation_history"] = conversation_history
            print(f"Using provided conversation history with {len(conversation_history)} messages")
        
        print(f"Sending request with thread_id: {thread_id}")
        
        # Try to get the current state to check if we're waiting for a category
        try:
            current_state = _graph.get_state(thread_id)
            print(f"Current state for thread {thread_id}: {current_state}")
            
            # If we have conversation history in the state, use it
            if current_state and "conversation_history" in current_state:
                print(f"Found conversation history in state with {len(current_state['conversation_history'])} messages")
                
            if current_state and current_state.get("category_slot") == "pending":
                print(f"Detected pending category slot, starting from slot_filler_router")
                # Start from the slot_filler_router node instead of the entry point
                result = await _graph.ainvoke(
                    input=input_data,
                    config={
                        "configurable": {"thread_id": thread_id},
                        "start_from_node": "slot_filler_router"  # Start from this node
                    }
                )
            else:
                # Normal flow starting from the entry point
                result = await _graph.ainvoke(
                    input=input_data,
                    config={"configurable": {"thread_id": thread_id}}
                )
                
            # Debug: Print the updated state after processing
            updated_state = _graph.get_state(thread_id)
            print(f"Updated state for thread {thread_id} after processing: {updated_state}")
        except Exception as e:
            print(f"Error getting state: {str(e)}, proceeding with normal flow")
            # If we can't get the state, just use the normal flow
            result = await _graph.ainvoke(
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
    # Use the singleton graph instance
    input_data = {"question": HumanMessage(content="tell me more about Building Materials & Hardware ranges?")}
    
    # Generate a unique thread_id for CLI mode
    thread_id = str(uuid.uuid4())
    print(f"Using thread_id: {thread_id}")
    
    result = asyncio.run(_graph.ainvoke(
        input=input_data,
        config={"configurable": {"thread_id": thread_id}}
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
        print(f"Created new thread_id: {st.session_state.thread_id}")
    else:
        print(f"Using existing thread_id: {st.session_state.thread_id}")
    
    # Initialize conversation history in session state if it doesn't exist
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        print("Initialized empty conversation history in session state")
    
    # Display chat messages
    st.title("🤖 Chatbot Assistant")
    st.write("Ask me anything about Building Materials & Hardware or other topics!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Update conversation history in session state
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        print(f"Added user message to conversation history. Total: {len(st.session_state.conversation_history)}")
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(get_chatbot_response(
                    prompt, 
                    st.session_state.thread_id,
                    st.session_state.conversation_history
                ))
                st.markdown(response, unsafe_allow_html=True)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update conversation history with assistant response
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        print(f"Added assistant response to conversation history. Total: {len(st.session_state.conversation_history)}")
        
        # Keep conversation history to a reasonable size (last 10 exchanges)
        if len(st.session_state.conversation_history) > 20:
            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
            print("Trimmed conversation history to last 20 messages")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run the CLI version
        asyncio.run(run_chatbot())
    else:
        # Run the Streamlit version by default
        run_streamlit_chatbot()
