# --- Imports ---
import os
import getpass
import pandas as pd
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import (
    MessagesState,
    START,
    END,
    StateGraph
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
    get_buffer_string
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain & LangGraph
from langgraph.graph import (
    MessagesState,
    START,
    END,
    StateGraph
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
    get_buffer_string
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Environment Setup ---
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mock-RAG"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Load CSVs ---
df_training_logs = pd.read_csv("../data/employee_training_logs.csv")
df_department_trainings = pd.read_csv("../data/department_trainings.csv")
df_updated_ecn_logs = pd.read_csv("../data/ecn_log.csv")

# --- Data Retrieval Tools ---
def retrieve_training_logs(question: str) -> str:
    print(f"[TOOL] Received query: {question}")

    rows = [
        f"Employee: {row['Employee']}\n"
        f"Department: {row['Department']}\n"
        f"Training: {row['Training']}\n"
        f"ECN: {row['ECN']}\n"
        f"Date: {row['Date']}"
        for _, row in df_training_logs.iterrows()
    ]
    text_block = "\n\n".join(rows[:100])

    system = SystemMessage(content="You are a helpful assistant. Filter and return only the rows relevant to the user's query.")
    human = HumanMessage(content=f"User query: {question}\n\nHere are the training logs:\n\n{text_block}")
    
    response = llm.invoke([system, human])
    return response.content or "No results found."

def retrieve_department_trainings(question: str) -> str:
    print(f"[TOOL] Received query: {question}")

    rows = [
        f"Department: {row['Department']}\n"
        f"Required Trainings: {row['Required Trainings']}"
        for _, row in df_department_trainings.iterrows()
    ]
    text_block = "\n\n".join(rows[:100])

    system = SystemMessage(content="You are a helpful assistant. Filter and return only the rows relevant to the user's query.")
    human = HumanMessage(content=f"User query: {question}\n\nHere are the departmental training programs:\n\n{text_block}")
    
    response = llm.invoke([system, human])
    return response.content or "No results found."

def retrieve_updated_ecn_logs(question: str) -> str:
    print(f"[TOOL] Received query: {question}")

    rows = [
        f"Training: {row['Training']}\n"
        f"Department: {row['Department']}\n"
        f"ECN: {row['ECN']}\n"
        f"Update Date: {row['Update Date']}\n"
        f"Description: {row['Description']}"
        for _, row in df_updated_ecn_logs.iterrows()
    ]
    text_block = "\n\n".join(rows[:100])

    system = SystemMessage(content="You are a helpful assistant. Filter and return only the rows relevant to the user's query.")
    human = HumanMessage(content=f"User query: {question}\n\nHere are the training ECN update logs:\n\n{text_block}")
    
    response = llm.invoke([system, human])
    return response.content or "No results found."

# --- Tool Definitions ---
retriever_tool_training_logs = Tool.from_function(
    func=retrieve_training_logs,
    name="retrieve_training_logs",
    description="Answer questions about employee training logs. Includes employee name, department, date, training completed, and ECN version."
)

retriever_tool_department_trainings = Tool.from_function(
    func=retrieve_department_trainings,
    name="retrieve_department_trainings",
    description="Answer questions about the departments' required trainings. Includes department name and necessary trainings."
)

retriever_tool_updated_ecn_logs = Tool.from_function(
    func=retrieve_updated_ecn_logs,
    name="retrieve_updated_ecn_logs",
    description="Answer questions about the ECN logs for trainings. Includes training, department, ECN version, update date, and description."
)

tools = [
    retriever_tool_training_logs,
    retriever_tool_department_trainings,
    retriever_tool_updated_ecn_logs
]

# --- State and Output Models ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class GradeOutput(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# --- LangGraph Nodes ---
def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    # Extract messages
    messages = state["messages"]
    last_message = messages[-1]
    docs = last_message.content
    question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

    llm_with_tool = llm.with_structured_output(GradeOutput)

    system = SystemMessage(content="""You are a grader assessing if all the information needed to answer the user question is present in the retrieved documents.
    If the document contains keyword(s) or semantic meaning related to the user question, it is relevant.
    Ensure all the documents and information required to answer a question have been retrieved.
    Do not make assumptions or provide any information that is not found in the relevant information.
    Give a binary score 'yes' or 'no' to indicate.""")

    human = HumanMessage(content=f"Here is the retrieved documents:\n\n{docs}\n\nHere is the user question:\n\n{question}")

    result = llm_with_tool.invoke([system, human])
    return "generate" if result.binary_score.lower() == "yes" else "agent"


def agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    agent_model = llm.bind_tools(tools)

    system = SystemMessage(content="""
    You are a company assistant that helps analyze employee training records across departments. You answer questions about:

    1. Employee Training Logs: This includes each employee's name, department, training completed, the ECN (Engineering Change Notice) version they trained with, and the date it was completed.

    2. Department Required Trainings: This lists the required training procedures for each department.

    3. ECN Logs for Trainings (very important): These list updates to training procedures, including the training name, updated ECN number, date of update, the department, and a short description of what changed. 
    This tells you whether the training the employee completed is outdated or not. Always consider ECN logs when the question involves update status or version comparison.

    When deciding how to answer a question, you may use one or more tools above to retrieve relevant information. Use tools if the current context is not enough.
    """)

    full_messages = [system] + messages  # Prepend system message

    response = agent_model.invoke(full_messages)
    return {"messages": [response]}

def generate(state: AgentState):
    messages = state["messages"]
    question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

    system = SystemMessage(content="""
    You are a helpful assistant answering a user's question about the company's training procedures.

    You have already retrieved information using tools. Now your job is to **answer the question strictly using the retrieved content only** — no assumptions or hallucinations.

    Always cross-reference those two sources when relevant. Be accurate and cautious — cite retrieved facts.
    """)

    human = HumanMessage(content=f"Here is the retrieved information:\n\n{messages}\n\nHere is the user question:\n\n{question}")

    response = llm.invoke([system, human])
    return {"messages": [response]}

# --- LangGraph Builder ---
builder = StateGraph(AgentState)

# Nodes
builder.add_node("agent", agent)
builder.add_node("retrieve", ToolNode(tools))
builder.add_node("generate", generate)

# Edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
builder.add_conditional_edges("retrieve", grade_documents, ["agent", "generate"])
builder.add_edge("generate", END)

graph = builder.compile()
