import os 
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner,RunContextWrapper,TResponseInputItem, OpenAIChatCompletionsModel, AsyncOpenAI,set_tracing_disabled, OutputGuardrailTripwireTriggered, output_guardrail,GuardrailFunctionOutput
from typing import List
load_dotenv()

set_tracing_disabled(disabled=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

provider = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL =OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)
class CountryOutput(BaseModel):
    is_country_allowed: bool
    country_name: str   
    reasoning: str
    answer: str

country_guardrail_agent = Agent(
    name="Check Country Guardrail Agent Check",
    instructions="You only allowed to answer questions related to Pakistan.",
    output_type=CountryOutput,
    model=MODEL
)

result  = Runner.run_sync(country_guardrail_agent, input="What is the capital of Pakistan?")
print(result.final_output.model_dump())  

@output_guardrail
async def output_guardrail_check(ctx:RunContextWrapper[None], agent: Agent, input: str | List[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(country_guardrail_agent, input,context=ctx.context)

    # print("[**Guardrails Response**]:\n", result.final_output,"\n\n")
   
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered= result.final_output.is_country_allowed)

agent = Agent(
    name="Country Guardrail Agent",
    instructions="You only allowed to answer questions related to Pakistan.",
    output_type=CountryOutput,
    model=MODEL
)

try:
        result = Runner.run_sync(agent, input="What is the capital?")
        print("Guradrails Response don't triggered")
        print(result.final_output)
except OutputGuardrailTripwireTriggered as e:
        print("Gruadrails Response triggered","\n", e)