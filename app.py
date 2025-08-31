from chatbot.graph import GraphConfig
from langchain_core.messages import HumanMessage
from typing import Dict, Any
import asyncio

async def run_chatbot():
    # Initialize the graph
    graph = GraphConfig.create_graph({})
    
    # Prepare input data with HumanMessage
    input_data = {"question": HumanMessage(content="tell me more about kitchen ranges?")}
    
    # Run the graph with invoke
    result = await graph.ainvoke(
        input=input_data,
        config={"configurable": {"thread_id": 2}}
    )
    
    # Print the final result
    if hasattr(result, "messages") and result.messages:
        last_message = result.messages[-1]
        if hasattr(last_message, "type") and last_message.type == "ai":
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    # Run the chatbot
    asyncio.run(run_chatbot())
