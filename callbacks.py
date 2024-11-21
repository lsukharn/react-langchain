from typing import Any, Dict
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        print(f"****Prompt to the LLM was: \n{prompts[0]}")
        print("*****")
        # return super().on_llm_start(serialized, prompts, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f"****Response from the LLM: \n{response.generations[0][0].text}")
        print("*****")
        # return super().on_llm_end(response, **kwargs)
