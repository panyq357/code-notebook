import os

from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class HelloAgentsLLM:

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model_id: Optional[str] = None, timeout: Optional[float] = None):
        
        base_url = base_url or os.getenv("LLM_API_BASE")
        api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_id = model_id or os.getenv("LLM_MODEL_ID")

        if not all([self.model_id, base_url, api_key]):
            raise ValueError("Base url, API key and model ID must be given, or defined as enviroment variables (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID).")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: list[dict[str, str]], temperature: float = 0) -> Optional[str]:
        print(f"Invoking {self.model_id} model ...")
        try:
            response = self.client.chat.completions.create(
                model = self.model_id,  # type:ignore
                messages = messages,  # type: ignore
                temperature = temperature,
                stream = True
            )
            print("Model {self.model_id} connected.")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as e:
            print(f" Error when invoking {self.model_id}: {e}")
            return None
