from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
# Mandatory model and config
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


app = FastAPI(title="OSS Note 1804812 Assessment & Remediation Prompt")

# ---- Strict input models ----
class select_item(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: Optional[str] = None

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def no_none(cls, v):
        return [x for x in v if x]

class NoteContext(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    mb_txn_usage: List[select_item] = Field(default_factory=list)

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def no_none(cls, v):
        return [x for x in v if x]

# ---- Planner summary ----
def summarize_context(ctx: NoteContext) -> dict:
    return {
        "system": ctx.system_name,
        "EHP_level": ctx.enhancement_pack,
        "environment": ctx.environment,
        "detected_obsolete_transactions": ctx.detected_transactions,
    }

# ---- LangChain prompt ----
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 1804812 who outputs strict JSON only."

USER_TEMPLATE = """
You are evaluating a system context related to SAP OSS Note 1804812 (MB* obsolescence). We provide:
- system context
- list of detected MB* transactions used in code

Your job:
1) Provide a concise **assessment**:
   - Indicate the risk: MB* is obsolete, warning may appear daily in non‑production.
   - Impact: user confusion, technical debt, eventual failure in future releases.
   - Recommend: use MIGO instead.

2) Provide an actionable **LLM remediation prompt**:
   - Reference system and environment.
   - Ask to search for uses of MB01, MB02, MB03, MB04, MB05, ΜΒΘΑ, MB11,
    MB1A, MB18, MBC, MB31, MBNL, MBRL, MBSF,
    MBSL, MBST, MBSU in code, replace them with MIGO equivalents.
   - Ensure same behavior, correct T007 etc.
   - Require output JSON with keys: original_code_snippet, remediated_code_snippet, changes[] (line/before/after/reason).

Return ONLY strict JSON:
{{
  "assessment": "<concise note 1804812 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context:
{context_json}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess(ctx: NoteContext):
    ctx_json = json.dumps(summarize_context(ctx), ensure_ascii=False, indent=2)
    return chain.invoke({"context_json": ctx_json,"pgm_name": ctx.pgm_name,
    "inc_name": ctx.inc_name,
    "type": ctx.type,
    "name": ctx.name})

@app.post("/assess-1804812")
def assess_note_context(ctx: NoteContext):
    try:
        result = llm_assess(ctx)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    return result

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
