import os 
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import (Agent, 
                    Runner, 
                    AsyncOpenAI, 
                    OpenAIChatCompletionsModel, 
                    set_tracing_disabled,
                    RunContextWrapper,
                    TResponseInputItem,
                    
                    GuardrailFunctionOutput,
                    InputGuardrailTripwireTriggered,
                    OutputGuardrailTripwireTriggered,
                    input_guardrail,
                    output_guardrail
                    )

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

class ChemistryQuestion(BaseModel):
    is_chemistry_question: bool
    reasoning: str
    answer :str

guardrail_agent = Agent(
    name="Check Chemistry Guardrail Agent",
    instructions="Check if the question is related to chemistry and provide reasoning.",
    output_type=ChemistryQuestion,
    model=MODEL
)

# result  = Runner.run_sync(guardrail_agent, input="What a formula for water?")
# print(result.final_output.is_chemistry_question)  # Should be True
# print(result.final_output.reasoning)  # Should contain reasoning
# print(result.final_output.answer)  # Should contain the answer, e.g., "H2O"


@input_guardrail
async def chemistry_gruardrail(ctx:RunContextWrapper[None], agent:Agent, input:str|List[TResponseInputItem])-> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input,context=ctx.context)

    print("[**Guardrails Response**]:\n", result.final_output)
    return GuardrailFunctionOutput(output_info=result.final_output,tripwire_triggered=result.final_output.is_chemistry_question)


chemistry_agent = Agent(
    name="Chemistry Question Agent",
    instructions="Answer chemistry questions. Help the user/students with chemistry-related queries.",
    input_guardrails=[chemistry_gruardrail],
    model=MODEL
)

try:
        result = Runner.run_sync(chemistry_agent, input="What is the formula for water?")
        print("Guradrails Response don't triggered")
        print(result.final_output)
except InputGuardrailTripwireTriggered as e:
        print("Gruadrails Response triggered","\n", e)
