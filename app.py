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
    st.set_page_config(
        page_title="PlaceMakers Assistant", 
        page_icon="🏗️",
        layout="centered"
    )
    
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
    
    # Initialize thinking status
    if "is_thinking" not in st.session_state:
        st.session_state.is_thinking = False
    
    # Custom CSS for the PlaceMakers design
    st.markdown("""
        <style>
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container styling */
        .stApp {
            background-color: #e8eaed;
        }
        
        /* Center the main content */
        .main .block-container {
            max-width: 750px !important;
            padding-top: 50px;
            padding-bottom: 50px;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        
        /* Wrapper for the entire chat UI */
        .chat-wrapper {
            max-width: 750px !important;
            margin: 0 auto !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Chat container */
        .chat-container {
            background-color: white;
            border-radius: 8px 8px 0 0;
            box-shadow: none;
            overflow: hidden;
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            max-width: 750px !important;
        }
        
        /* Header styling */
        .chat-header {
            background-color: #0d5a8f;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 18px;
            font-weight: 500;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        }
        
        /* Chat messages area */
        .chat-messages {
            min-height: 400px;
            max-height: 500px;
            overflow-y: auto;
            padding: 20px;
            background-color: white;
            margin-bottom: 0 !important;
        }
        
        /* User message styling */
        .user-message {
            background-color: #0d5a8f;
            color: white;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 10px 0;
            max-width: 70%;
            margin-left: auto;
            text-align: left;
            word-wrap: break-word;
            display: block;
        }
        
        /* Assistant message styling */
        .assistant-message {
            background-color: #f1f3f4;
            color: #202124;
            padding: 12px 16px;
            border-radius: 18px;
            margin: 10px 0;
            max-width: 70%;
            margin-right: auto;
            text-align: left;
            word-wrap: break-word;
            display: block;
        }
        
        .assistant-label {
            font-size: 12px;
            color: #5f6368;
            margin-bottom: 4px;
            font-weight: 500;
        }
        
        /* Input area inside container */
        .input-container {
            padding: 15px 20px;
            border-top: 1px solid #e0e0e0;
            background-color: white;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .custom-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #dadce0;
            border-radius: 24px;
            font-size: 14px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            outline: none;
            resize: none;
            min-height: 20px;
            max-height: 100px;
        }
        
        .custom-input:focus {
            border-color: #0d5a8f;
        }
        
        .send-button {
            background-color: #0d5a8f;
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 18px;
        }
        
        .send-button:hover {
            background-color: #094a73;
        }
        
        /* Hide the default chat input */
        section[data-testid="stChatInput"] {
            display: none !important;
        }
        
        /* Style the form container */
        .stForm {
            border: none !important;
            padding: 15px 20px !important;
            background-color: white !important;
            border-top: 1px solid #e0e0e0 !important;
            margin-top: -8px !important;
            border-radius: 0 0 8px 8px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
            max-width: 750px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        
        /* Style form input */
        .stTextInput > div > div > input {
            border-radius: 24px !important;
            border: 1px solid #dadce0 !important;
            padding: 12px 16px !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #0d5a8f !important;
            box-shadow: none !important;
        }
        
        .stTextInput > label {
            display: none !important;
        }
        
        /* Style submit button */
        .stFormSubmitButton > button {
            background-color: #0d5a8f !important;
            color: white !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            min-height: 40px !important;
            padding: 0 !important;
            border: none !important;
            font-size: 18px !important;
        }
        
        .stFormSubmitButton > button:hover {
            background-color: #094a73 !important;
        }
        
        .stFormSubmitButton > button p {
            font-size: 18px !important;
        }
        
        /* Status message area */
        .status-message {
            padding: 10px 20px;
            background-color: white;
            border-top: 1px solid #e0e0e0;
            color: #202124;
            font-size: 14px;
            font-style: italic;
            text-align: center;
            min-height: 20px;
        }
        
        /* Hide default streamlit chat UI elements we don't want */
        .stChatMessage {
            background-color: transparent !important;
            padding: 0 !important;
        }
        
        div[data-testid="stChatMessageContent"] {
            background-color: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create a container for the chat UI
    chat_container = st.container()
    
    with chat_container:
        # Build the entire chat UI as a single HTML block
        chat_html = '<div class="chat-wrapper"><div class="chat-container"><div class="chat-header">PlaceMakers Shopping Assistant</div><div class="chat-messages" id="chat-messages">'
        
        # Add messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                chat_html += f'<div class="user-message">{message["content"]}</div>'
            else:
                chat_html += f'<div class="assistant-message"><div class="assistant-label">TFG</div><div>{message["content"]}</div></div>'
        
        # Close chat messages and add status area
        chat_html += '</div>'
        
        # Add status message area
        status_text = "TFG is thinking..." if st.session_state.is_thinking else ""
        chat_html += f'<div class="status-message">{status_text}</div>'
        
        # Close the HTML structure
        chat_html += '</div></div>'
        
        # Render the complete HTML
        st.markdown(chat_html, unsafe_allow_html=True)
    
    # Form for input in a separate container
    form_container = st.container()
    
    with form_container:
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            with col1:
                prompt = st.text_input("", placeholder="Ask something...", label_visibility="collapsed", key="user_input")
            with col2:
                submit = st.form_submit_button("➤", use_container_width=True)
    
    if submit and prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Update conversation history in session state
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        print(f"Added user message to conversation history. Total: {len(st.session_state.conversation_history)}")
        
        # Set thinking status to true
        st.session_state.is_thinking = True
        st.rerun()  # Rerun to show thinking status
        
    # Process the response if thinking
    if st.session_state.is_thinking:
        # Get assistant response
        response = asyncio.run(get_chatbot_response(
            st.session_state.messages[-1]["content"], 
            st.session_state.thread_id,
            st.session_state.conversation_history
        ))
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update conversation history with assistant response
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        print(f"Added assistant response to conversation history. Total: {len(st.session_state.conversation_history)}")
        
        # Keep conversation history to a reasonable size (last 10 exchanges)
        if len(st.session_state.conversation_history) > 20:
            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
            print("Trimmed conversation history to last 20 messages")
        
        # Reset thinking status
        st.session_state.is_thinking = False
        
        # Rerun to update the UI
        st.rerun()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run the CLI version
        asyncio.run(run_chatbot())
    else:
        # Run the Streamlit version by default
        run_streamlit_chatbot()
