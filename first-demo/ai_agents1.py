# gemini agent -- cuz it free

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)



class AgentState(TypedDict):
    messages: List[str]
    user_model: str
    streaming: bool
    agent : any



class Agent:
    def __init__(self, user_model: str = "gemini-1.5-flash", streaming: bool = False):
        self.user_model = user_model
        self.streaming = streaming
        self.model = GenerativeModel(user_model)
        self.chat = self.model.start_chat(history=[])
    
    def enable_streaming(self):
        self.streaming = True


def user_input_node(state: AgentState) -> AgentState:
    """Node to get user input"""
    user_input = input("\nYou: ")
    state['messages'].append(f"User: {user_input}")

    return state

def create_agent(state:AgentState) -> AgentState:
    '''create the agent if none exists'''
    if not state['agent']:
        state['user_model']  = "gemini-1.5-flash"
        state['streaming'] = False
        state['agent'] = Agent(user_model=state['user_model'], streaming=state['streaming'])
    else:
        pass
    
    return state

def ask_node(state: AgentState) -> AgentState:
    """Node to process agent response"""
    last_message = state['messages'][-1]
    if last_message.startswith("User: "):
        prompt = last_message[6:]
    else:
        prompt = last_message

    response = state['agent'].chat.send_message(prompt, stream=state['streaming'])
    text = response.candidates[0].content.parts[0].text
    state['messages'].append(f"Agent: {text}")


    return state

def should_continue(state: AgentState) -> str:
    """Conditional edge to determine if we should continue chatting"""
    if not state['messages']:
        return "continue"
    
    last_message = state['messages'][-1]
    
    if last_message.startswith("User: "):
        user_text = last_message[6:].strip().lower()
        if user_text in ['quit', 'exit', 'bye', 'stop']:
            print("Ending....")
            return "end"
    
    return "continue"


def print_res_node(state:AgentState) -> AgentState:
    '''this node print the result of the querry'''
    last_res = state['messages'][-1]
    if last_res.startswith("Agent: "):
        print(last_res)

    return state

def end_node(state: AgentState) -> AgentState:
    print("Agent: Goodbye 👋")
    return state

bot_graph = StateGraph(AgentState)

bot_graph.add_node("input", user_input_node)
bot_graph.add_node('make_agent', create_agent)
bot_graph.add_node('ask', ask_node)
bot_graph.add_node('output', print_res_node)
bot_graph.add_node("router", lambda state:state)
bot_graph.add_node("end_node", end_node)

bot_graph.add_conditional_edges(
    'router',
    should_continue,
    {
        'end' : 'end_node',
        'continue' : 'ask'
    }
)

bot_graph.add_edge(START, 'make_agent')
bot_graph.add_edge('make_agent', 'input')
bot_graph.add_edge('input', 'router')
bot_graph.add_edge('ask','output')
bot_graph.add_edge('output', 'input')
bot_graph.add_edge("end_node", END)

if __name__ == '__main__':
    agent = bot_graph.compile()
    print("\nType 'quit', 'stop', 'bye' or 'exit' to kill the process\n")
    initial_state = {
        'messages': ['Hi there'],
        'user_model': 'gemini-2.5-flash',
        'streaming': False,
        'agent': None
    }
    agent.invoke(initial_state)




