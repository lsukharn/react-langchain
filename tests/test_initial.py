from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
import main_agent
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama

class OllamaAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"
    
custom_model = ChatOllama(
    temperature=0.1,
    model="mistral"
)
ollama = OllamaAI(model=custom_model)


def test_agent():
    input = "What is the length of 'Akin' in characters?"
    actual_output = main_agent.run_agent(input)

    metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        model=ollama,
        # NOTE: you can only provide either criteria or evaluation_steps, and not both
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are not OK"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )

    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output="length is 4"
    )

    metric.measure(test_case)
    print(metric.score)
    print(metric.reason)