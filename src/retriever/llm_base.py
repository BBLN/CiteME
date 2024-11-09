import os
from langchain_openai import ChatOpenAI

from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig

DEFAULT_TEMPERATURE = 0.95

def get_model_by_name(model_name: str, temperature: float = DEFAULT_TEMPERATURE):
    if model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
        # return AzureOpenAI(model=model_name, temperature=temperature)
    if model_name.startswith("claude-"):
        return ChatAnthropic(
            temperature=temperature,
            model_name=model_name,
            timeout=60*10,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    if "phi" in model_name.lower():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        llm = HuggingFacePipeline.from_model_id(
            model_id="microsoft/Phi-3.5-mini-instruct",
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=False,
                return_full_text=False,
            ),
            model_kwargs={
                'quantization_config': quantization_config
            },
        )

        return ChatHuggingFace(llm=llm)
    if "llama" in model_name.lower() or "phi" in model_name.lower() or "mistral" in model_name.lower():
        return ChatTogether(
            # together_api_key="YOUR_API_KEY",
            temperature=temperature,
            model=model_name,
        )
    raise ValueError(f"Model {model_name} not found")
