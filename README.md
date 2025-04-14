Simple Retrieval-Augmented Generation system built with **LangGraph** and **LangChain**, designed to assist with employee training compliance in a manufacturing setting. Built to learn LangGraph and experiment with its functionalities. In future would like to experiment with different memory management techniques, parallel tool calls, HITL interuptions to the workflow, and more robust file loading. 

- LangGraph agent flow with conditional routing and tool use
- Custom tools for querying csv documents
- Grading logic to decide if information is sufficient before answering
- Support for OpenAI GPT models

### 1. Clone the repo

### 2. Install Dependencies 
pip install -r requirements.txt

### 3. Set up enviorment 
Create a .env file in studio directory with...

OPENAI_API_KEY=your-key

LANGCHAIN_API_KEY=your-key

LANGCHAIN_TRACING_V2=true

LANGCHAIN_PROJECT=mock-RAG

### 4. Open and Run with LangGraph Studio 
Follow steps from LangGraph for opening project in studio. In general, activate enviroment in terminal, navigate to "studio" folder, run command "lanngraph dev", enter LangChain API key.
