from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

app = FastAPI(title="OSS Note 1804812 Assessment")

class NoteContext(BaseModel):
    table: Optional[str] = None
    target_type: Optional[str] = None
    target_name: Optional[str] = None
    used_fields: List[str] = []
    suggested_fields: List[str] = []
    suggested_statement: Optional[str] = None
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    unit_type: Optional[str] = None
    unit_name: Optional[str] = None
    system_name: Optional[str] = None
    enhancement_pack: Optional[str] = None
    environment: Optional[str] = None
    detected_transactions: List[str] = Field(default_factory=list)

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def remove_none(cls, v): return [x for x in v if x]

def summarize_context(ctx: NoteContext) -> dict:
    return {
        "system": ctx.system_name,
        "EHP_level": ctx.enhancement_pack,
        "environment": ctx.environment,
        "detected_obsolete_transactions": ctx.detected_transactions,
    }

SYSTEM_MSG = "You are an ABAP reviewer for SAP Note 1804812. Respond only in strict JSON."

USER_TEMPLATE = """
System context and metadata are below.

Return ONLY strict JSON with 2-3 sentence assessment and a one-paragraph LLM prompt:
{{
  "assessment": "<short summary>",
  "llm_prompt": "<brief prompt>"
}}

Program: {pgm_name}, Include: {inc_name}, Type: {unit_type}, Name: {unit_name}
System: {context_json}
"""

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_MSG), ("user", USER_TEMPLATE)])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
chain = prompt | llm | JsonOutputParser()

def llm_assess(ctx: NoteContext):
    return chain.invoke({
        "context_json": json.dumps(summarize_context(ctx), ensure_ascii=False),
        "pgm_name": ctx.pgm_name,
        "inc_name": ctx.inc_name,
        "unit_type": ctx.unit_type,
        "unit_name": ctx.unit_name
    })

@app.post("/assess-1804812")
def assess(ctx: NoteContext):
    try:
        return llm_assess(ctx)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.get("/health")
def health(): return {"ok": True, "model": OPENAI_MODEL}
