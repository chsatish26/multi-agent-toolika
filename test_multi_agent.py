"""
Test script for the multi-agent system.
This can be used to verify the system works correctly.
"""

import uuid
from multi_agent_langsmith import graph
from langchain_core.messages import HumanMessage
from langgraph.types import Command

def test_basic_flow():
    """Test the basic flow of the multi-agent system."""
    
    # Generate a unique thread ID
    thread_id = str(uuid.uuid4())
    
    # Test with a customer that has ID verification
    question = "My customer ID is 1. What's my most recent purchase? What albums do you have by the Rolling Stones?"
    
    config = {
        "configurable": {
            "thread_id": thread_id, 
            "user_id": "1"
        }
    }
    
    print("Testing multi-agent system...")
    print(f"Question: {question}")
    print("-" * 50)
    
    try:
        # Invoke the graph
        result = graph.invoke({
            "messages": [HumanMessage(content=question)]
        }, config=config)
        
        print("Final result:")
        for message in result["messages"]:
            print(f"Type: {type(message).__name__}")
            print(f"Content: {message.content}")
            print("-" * 30)
            
        return True
        
    except Exception as e:
        print(f"Error during execution: {e}")
        return False

def test_interrupt_flow():
    """Test the interrupt flow when customer ID is not provided."""
    
    # Generate a unique thread ID
    thread_id = str(uuid.uuid4())
    
    question = "I'd like to know about my recent purchases and find some rock music."
    
    config = {
        "configurable": {
            "thread_id": thread_id, 
            "user_id": "10"
        }
    }
    
    print("\nTesting interrupt flow...")
    print(f"Question: {question}")
    print("-" * 50)
    
    try:
        # First invocation - should interrupt for customer verification
        result = graph.invoke({
            "messages": [HumanMessage(content=question)]
        }, config=config)
        
        print("After first invocation (should ask for ID):")
        print(f"Last message: {result['messages'][-1].content}")
        
        # Resume with customer ID
        result = graph.invoke(
            Command(resume="My customer ID is 10"), 
            config=config
        )
        
        print("\nAfter providing customer ID:")
        print(f"Final message: {result['messages'][-1].content}")
        
        return True
        
    except Exception as e:
        print(f"Error during interrupt flow: {e}")
        return False

if __name__ == "__main__":
    print("Running multi-agent system tests...")
    
    success1 = test_basic_flow()
    success2 = test_interrupt_flow()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")