from chatbot.graph import GraphConfig
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any, List
import asyncio
import streamlit as st
import uuid
from pathlib import Path
import html
import base64

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
            
            # If we have conversation history in the state, use it instead of the one from session state
            # This ensures continuity in conversations across multiple requests
            if current_state and "conversation_history" in current_state:
                print(f"Found conversation history in state with {len(current_state['conversation_history'])} messages")
                # Use conversation history from state instead of from input
                # but only if state history is longer (more complete)
                if len(current_state['conversation_history']) > len(conversation_history or []):
                    input_data["conversation_history"] = current_state["conversation_history"]
                    print(f"Using conversation history from state instead of session")
                
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

def load_css() -> str:
    """Load CSS from external file."""
    css_path = Path(__file__).parent / "static" / "styles.css"
    with open(css_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_logo_as_base64() -> str:
    """Load logo SVG and convert to base64 data URI."""
    logo_path = Path(__file__).parent / "static" / "placemakers-new-logo.svg"
    with open(logo_path, 'rb') as f:
        logo_data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/svg+xml;base64,{logo_data}"

def load_favicon_as_base64() -> str:
    """Load favicon and convert to base64 data URI."""
    favicon_path = Path(__file__).parent / "static" / "placemakers-favicon.ico"
    with open(favicon_path, 'rb') as f:
        favicon_data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/x-icon;base64,{favicon_data}"

def load_html_template() -> str:
    """Load HTML template from external file."""
    html_path = Path(__file__).parent / "static" / "chat_template.html"
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

def run_streamlit_chatbot():
    """Run the chatbot with a Streamlit web interface."""
    # Get the path to the favicon
    favicon_path = Path(__file__).parent / "static" / "placemakers-favicon.ico"
    
    st.set_page_config(
        page_title="BuildMate - PlaceMakers Assistant", 
        page_icon=str(favicon_path),
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
    
    # Initialize button click state
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None
    
    # Load and apply CSS from external file
    css_content = load_css()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    
    # Create a container for the chat UI
    chat_container = st.container()
    
    # Load logo and favicon as base64
    logo_base64 = load_logo_as_base64()
    
    with chat_container:
        # Build messages HTML
        messages_html = ""
        if st.session_state.messages:
            for message in st.session_state.messages:
                # Escape user content to prevent HTML injection, but allow assistant HTML
                content = message["content"]
                if message["role"] == "user":
                    content = html.escape(content)
                    messages_html += f'<div class="user-message">{content}</div>'
                else:
                    # Assistant messages may contain HTML (like product cards), so don't escape
                    messages_html += f'<div class="assistant-message"><div class="assistant-label">BuildMate</div><div>{content}</div></div>'
        else:
            # Show welcome message when no messages
            from datetime import datetime
            current_time = datetime.now().strftime("%H:%M")
            messages_html = f'''<div class="welcome-container">
<div class="welcome-message">
Kia ora! Welcome to BuildMate, your 24/7 PlaceMakers expert. I'm here to help with your building projects, product searches, or order needs. What can I do for you today?
<div class="welcome-timestamp">{current_time}</div>
</div>
</div>'''
        
        # Get status text
        status_text = "BuildMate is thinking..." if st.session_state.is_thinking else ""
        
        # Load HTML template and replace placeholders
        html_template = load_html_template()
        chat_html = html_template.format(messages=messages_html, status=status_text, favicon=logo_base64)
        
        # Render the complete HTML
        st.markdown(chat_html, unsafe_allow_html=True)
    
    # Form for input in a separate container
    form_container = st.container()
    
    with form_container:
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            with col1:
                prompt = st.text_input("", placeholder="Ask about any products or projects", label_visibility="collapsed", key="user_input")
            with col2:
                submit = st.form_submit_button("➤", use_container_width=True, type="primary")
    
    # Show quick action buttons if no messages - positioned BELOW input form
    if not st.session_state.messages:
        button_container = st.container()
        with button_container:
            # Button 1: Find Products
            if st.button("📦 Find Products", key="btn_find_products", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Find Products"})
                st.session_state.conversation_history.append({"role": "user", "content": "Find Products"})
                st.session_state.is_thinking = True
                st.rerun()
            
            # Button 2: Get Project Advice
            if st.button("📘 Get Project Advice", key="btn_project_advice", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Get Project Advice"})
                st.session_state.conversation_history.append({"role": "user", "content": "Get Project Advice"})
                st.session_state.is_thinking = True
                st.rerun()
            
            # Button 3: Learn About Deals
            if st.button("🏷️ Learn About Deals", key="btn_deals", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Learn About Deals"})
                st.session_state.conversation_history.append({"role": "user", "content": "Learn About Deals"})
                st.session_state.is_thinking = True
                st.rerun()
    
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
