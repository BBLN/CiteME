import os
from langchain_openai import ChatOpenAI

from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftConfig, PeftModel
from unsloth import FastLanguageModel

DEFAULT_TEMPERATURE = 0.95

def get_model_by_name(model_name: str, peft_adapter=None, temperature: float = DEFAULT_TEMPERATURE, generation_kwargs=None):
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
        
        model_id="microsoft/Phi-3.5-mini-instruct"
        #tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_kwargs = {}
        load_in_4bit = None
        if '4bit' in model_name:
            model_kwargs['quantization_config'] = quantization_config
            load_in_4bit = True

        #model = AutoModelForCausalLM.from_pretrained(
        #    model_id,
        #    **model_kwargs)
        #if peft_adapter:
        #    model = PeftModel.from_pretrained(model, "../NLP-project/" + peft_adapter)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=f"../NLP-project/sft_outputs/{peft_adapter}" if peft_adapter else "unsloth/Phi-3.5-mini-instruct",
            load_in_4bit=load_in_4bit)
        FastLanguageModel.for_inference(model)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
            #top_p=0.6,
            return_full_text=False,
            **(generation_kwargs or {})
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        class ChatHuggingFacePhi(ChatHuggingFace):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def _to_chat_prompt(self, messages):
                if not messages:
                    raise ValueError("At least one HumanMessage must be provided!")

                messages_dicts = [self._to_chatml_format(m) for m in messages]

                return self.tokenizer.apply_chat_template(
                    messages_dicts, 
                    tokenize=False,
                    add_generation_prompt=True,
                    #continue_final_message=True,
                )
        return ChatHuggingFacePhi(llm=llm)
    if "llama" in model_name.lower() or "phi" in model_name.lower() or "mistral" in model_name.lower():
        return ChatTogether(
            # together_api_key="YOUR_API_KEY",
            temperature=temperature,
            model=model_name,
        )
    raise ValueError(f"Model {model_name} not found")
