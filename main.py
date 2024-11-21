from typing import List, Union
from dotenv import load_dotenv
from langchain.agents import tool, Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_ollama import ChatOllama
from langchain.agents.format_scratchpad.log import format_log_to_str
from callbacks import AgentCallbackHandler

load_dotenv()


@tool  # creates a lnagchain tool from the function
def get_text_length(text: str) -> int:
    """Return the length of a text by characters"""  # Used by LLM to understand the purpose of the function
    text = text.strip("'\n").strip(
        '"'
    )  # stripping non-alphanumeric chars to avoid parsing errors

    return len(text)


@tool
def get_request(r: str):
    """Return request object"""
    return r


@tool
def get_N_dollars(n: str):
    """Return N dollars"""
    return f"Here is {n} dollars"


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
        else:
            raise ValueError(f"Tool with {tool_name} not found")


if __name__ == "__main__":
    print("React")
    # print(get_text_langth("test text"))

    tools = [get_text_length, get_request, get_N_dollars]

    # chain of thought prompt, bc we are asking the LLM how is it thinking
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    # partial will add "tools" that already exists, "input" is added later
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # "stop" argument will tell the LLM to finish working as soon as it ouputs `\nObservation`
    # to prevent the LLM from producing halucinations. It concludes with Observation, but then it may start adding
    # stuff on its own, halucinate
    llm = ChatOllama(
        temperature=0.1,
        model="mistral",
        stop=["\nObservation", "Observation"],
        callbacks=[AgentCallbackHandler()],
    )

    # part 1: input plus thought generation
    # part 2: parsing of the LLM thought. Uses RegEx in `def parse()`
    intermediate_steps = (
        []
    )  # history of agent's actions is stored here. It gets updated after every iteration.
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # part 3: Execution
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of 'Akin' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(agent_step)

        if isinstance(agent_step, AgentFinish):
            break
        
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        # agent_step should be a text, but it's an object, so use format_log_to_str() to convert AgentAction to str
        intermediate_steps.append((agent_step, observation))

    # intermediate_steps = []
    # agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "How to get '10' dollars?",
    #                                                             "agent_scratchpad": intermediate_steps})
    # print(agent_step)
