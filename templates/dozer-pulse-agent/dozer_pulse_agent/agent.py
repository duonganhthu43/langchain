from langchain_community.utilities import DozerPulseWrapper
from langchain_community.agent_toolkits import DozerPulseToolkit
from langchain_community.agent_toolkits.dozer.base import create_dozer_agent, create_dozer_agent_simple
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple
from langchain.agents.agent_types import AgentType
import os

api_key = os.getenv("DOZER_API_KEY")
application_id = os.getenv("DOZER_APPLICATION_ID")
dozer = DozerPulseWrapper(api_key=api_key, application_id=application_id)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
toolkit = DozerPulseToolkit(dozer=dozer, llm=llm)

class AgentInput(BaseModel):
    input: str
    api_key: str
    application_id: int
    # langchain_tracing_v2: bool
    # langchain_api_key: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )
class AgentOutput(BaseModel):
    output: str
    
agent_executor = create_dozer_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION).with_types(
    input_type=AgentInput
)

