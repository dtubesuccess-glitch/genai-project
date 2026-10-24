from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# 1. Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# 2. Define custom tools
@tool
def get_weather(city: str) -> str:
    """Returns weather information for a given city."""
    return f"The weather in {city} is currently sunny and 25°C."

@tool
def calculate_budget(income: float, expenses: float) -> str:
    """Calculates monthly savings based on income and expenses."""
    savings = income - expenses
    return f"Your estimated monthly savings is ₹{savings:.2f}."

tools = [get_weather, calculate_budget]
tools_dict = {"get_weather": get_weather, "calculate_budget": calculate_budget}

llm_with_tools = llm.bind_tools(tools)

# 3. Maintain chat history
messages = [
    SystemMessage(content="You are a helpful, courteous AI assistant equipped with real-time tools.")
]

print("==================================================")
print("   Welcome to your GenAI Interactive Assistant!   ")
print("   (Type 'exit' or 'quit' anytime to stop)        ")
print("==================================================\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Assistant: Goodbye! Have a great day ahead.")
        break

    if not user_input.strip():
        continue

    # Add user message to memory
    messages.append(HumanMessage(content=user_input))

    # Get response from model
    response = llm_with_tools.invoke(messages)

    # Handle tool calls if triggered by the model
    if response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name in tools_dict:
                tool_output = tools_dict[tool_name].invoke(tool_args)
                print(f"\n[Tool Executed ({tool_name}) -> Output: {tool_output}]\n")
                
                # Pass tool output back to the LLM to form a final conversational answer
                messages.append(HumanMessage(content=f"Tool result: {tool_output}"))
                final_response = llm.invoke(messages)
                print(f"Assistant: {final_response.content}\n")
                messages.append(final_response)
    else:
        print(f"Assistant: {response.content}\n")
        messages.append(response)