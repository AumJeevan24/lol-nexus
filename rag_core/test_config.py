# rag_core/test_config.py
import os
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.model = ChatGroq(
            temperature=0,
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.model.ainvoke(prompt)).content

    def get_model_name(self):
        return "Groq Llama-3"

# Initialize the judge for import
groq_judge = GroqJudge()