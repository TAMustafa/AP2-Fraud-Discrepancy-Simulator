import os
import hmac
import hashlib
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

# Optional OpenAI LLM via LangChain
try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore

# --- Env & logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("ap2-sim")

# --- Canonicalization & HMAC signing (demo-grade, not production) ---
def canonicalize(data: Dict[str, Any]) -> bytes:
    """Return a canonical JSON byte representation.
    Ensures stable ordering and no extra whitespace.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def sign_mandate(mandate: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Sign a mandate using HMAC-SHA256 over canonical JSON body.
    The signature is demo-grade and includes minimal metadata.
    """
    body = dict(mandate)
    body.pop("signature", None)
    mac = hmac.new(key.encode("utf-8"), canonicalize(body), hashlib.sha256).hexdigest()
    mandate["signature"] = mac
    mandate["sig_alg"] = "HMAC-SHA256"
    return mandate

def verify_mandate(mandate: Dict[str, Any], key: str) -> bool:
    """Verify a signed mandate by recomputing the HMAC."""
    sig = mandate.get("signature", "")
    body = dict(mandate)
    body.pop("signature", None)
    body.pop("sig_alg", None)
    expected = hmac.new(key.encode("utf-8"), canonicalize(body), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)

# --- Keys per role ---
keys: Dict[str, str] = {
    "user": os.getenv("USER_SECRET", "u123"),
    "agent": os.getenv("AGENT_SECRET", "a123"),
    "merchant": os.getenv("MERCHANT_SECRET", "m123"),
    "cred": os.getenv("CREDENTIALS_SECRET", "c123"),
}

# --- Scenario toggle (global) ---
current_scenario: str = "Normal"  # kept for UI display/logging only
use_llm_agent: bool = False        # kept for UI display/logging only

# --- Role functions ---
def _new_id(prefix: str) -> str:
    """Create a simple unique ID using current time seconds.
    
    Args:
        prefix: Short prefix indicating mandate type.
    
    Returns:
        Deterministic-ish ID for demo purposes.
    """
    return f"{prefix}_{int(time.time()*1000)}"

def user_tool(scenario: str) -> Dict[str, Any]:
    """Create a signed intent mandate from the user.
    
    Returns:
        Signed intent mandate dict.
    """
    expiry = int(time.time()) + 60
    if scenario == "Expired Mandate":
        expiry = int(time.time()) - 10
    intent: Dict[str, Any] = {
        "mandate_type": "intent",
        "mandate_id": _new_id("intent"),
        "max_amount": 120,
        "category": "sneakers",
        "expiry": expiry,
        "created_at": int(time.time()),
    }
    return sign_mandate(intent, keys["user"])

def _get_llm() -> Optional[Any]:
    """Return a ChatOpenAI instance if available and configured, else None."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY not set; running without LLM")
        return None
    if ChatOpenAI is None:
        logger.warning("langchain-openai not installed; running without LLM")
        return None
    try:
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to initialize ChatOpenAI: %s", e)
        return None

def agent_tool(intent: Dict[str, Any], scenario: str, llm_agent: bool) -> Dict[str, Any]:
    """Create a signed cart mandate from the agent based on the intent.
    Optionally uses an LLM to choose item/price (demo behavior).
    """
    item = intent.get("category")
    price = 100
    proposed_item = item
    proposed_price = price

    if llm_agent:
        llm = _get_llm()
        if llm is not None:
            prompt = (
                "You are a shopping agent helping assemble a cart based on a user intent.\n"
                "Return ONLY a compact JSON object on a single line with keys 'item' and 'price'.\n"
                "Example: {\"item\": \"sneakers\", \"price\": 100}\n"
                "Constraints:\n"
                "- item SHOULD match the user's category for typical flows.\n"
                "- price SHOULD be a positive integer.\n"
                "Context: category=" + str(intent.get('category')) + ", max_amount=" + str(intent.get('max_amount')) + ", scenario=" + scenario + "."
            )
            try:
                resp = llm.invoke(prompt)
                # LangChain messages may expose .content as str or list of parts
                raw = getattr(resp, "content", "")
                if isinstance(raw, list) and raw:
                    content = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in raw])
                else:
                    content = raw if isinstance(raw, str) else ""
                # Extract JSON if wrapped in code fences
                text = content.strip()
                if text.startswith("```"):
                    text = text.strip('`')
                    if "\n" in text:
                        text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                try:
                    data = json.loads(text)
                except Exception:
                    data = {}
                # Sanitize LLM outputs
                proposed_item = str(data.get("item", item))
                proposed_price = data.get("price", price)
                try:
                    proposed_price = int(proposed_price)
                except Exception:
                    proposed_price = price
            except Exception as e:  # pragma: no cover - keep deterministic fallback
                logger.warning("LLM selection failed, using defaults: %s", e)

    # Enforce category fidelity at the agent step for ALL scenarios
    item = intent.get("category", proposed_item)

    # Price policy by scenario
    max_amount = int(intent.get("max_amount", 120))
    if scenario == "Bad Agent":
        # Ensure it exceeds the cap to trigger the intended discrepancy
        price = max(int(proposed_price), max_amount + 30)
    else:
        # Keep price within allowed bounds
        price = min(max(1, int(proposed_price)), max_amount)

    cart: Dict[str, Any] = {
        "mandate_type": "cart",
        "mandate_id": _new_id("cart"),
        "parent_id": intent["mandate_id"],
        "item": item,
        "price": price,
        "currency": intent.get("currency", "USD"),
        "created_at": int(time.time()),
        "agent_id": "agent-001",
    }
    cart = sign_mandate(cart, keys["agent"])
    logger.info("Cart created: %s", {k: cart[k] for k in ["mandate_id", "parent_id", "item", "price"]})
    return cart

def merchant_tool(cart: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Create a signed payment mandate from the merchant based on the cart."""
    payment: Dict[str, Any] = {
        "mandate_type": "payment",
        "mandate_id": _new_id("payment"),
        "parent_id": cart["mandate_id"],
        "item": cart["item"],
        "amount": cart["price"],
        "currency": cart.get("currency", "USD"),
        "created_at": int(time.time()),
        "merchant_id": "merchant-001",
    }
    if scenario == "Bad Merchant":
        payment["item"] = "sandals"
    payment = sign_mandate(payment, keys["merchant"])
    logger.info("Payment created: %s", {k: payment[k] for k in ["mandate_id", "parent_id", "item", "amount"]})
    return payment

def credentials_tool(payment: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    """Simulate a credentials step that could tamper with the signature."""
    if scenario == "Invalid Signature":
        payment["signature"] = "FAKE"
    return payment

# --- Validation logic ---
def validate_flow(intent: Dict[str, Any], cart: Dict[str, Any], payment: Dict[str, Any]) -> List[str]:
    """Validate the chain of mandates and report discrepancies.
    
    Returns:
        List of human-readable validation results.
    """
    results: List[str] = []

    # Structural checks
    if cart.get("parent_id") != intent.get("mandate_id"):
        results.append("❌ Cart parent_id does not reference Intent")
    if payment.get("parent_id") != cart.get("mandate_id"):
        results.append("❌ Payment parent_id does not reference Cart")

    # Business rule checks
    if cart.get("price", 0) > intent.get("max_amount", 0):
        results.append("❌ Cart exceeds max amount")
    if cart.get("item") != intent.get("category"):
        results.append("❌ Agent chose item not in intent category")
    if payment.get("item") != cart.get("item"):
        results.append("❌ Merchant swapped item")
    if payment.get("amount") != cart.get("price"):
        results.append("❌ Payment amount does not match cart price")
    if intent.get("expiry", 0) < time.time():
        results.append("❌ Intent expired")
    if payment.get("currency") != cart.get("currency"):
        results.append("❌ Currency mismatch between cart and payment")

    # Signature checks
    if not verify_mandate(intent, keys["user"]):
        results.append("❌ Invalid user signature on intent")
    if not verify_mandate(cart, keys["agent"]):
        results.append("❌ Invalid agent signature on cart")
    if not verify_mandate(payment, keys["merchant"]):
        results.append("❌ Invalid merchant signature on payment")

    if not results:
        results = ["✅ Transaction valid"]
    return results

# --- Simulation orchestrator ---
def run_simulation(scenario: str, llm_agent: bool) -> Tuple[Dict[str, Any], str, str]:
    """Run the end-to-end demo flow for a chosen scenario.

    Args:
        scenario: One of the dropdown scenario labels.

    Returns:
        A tuple of (mandates_dict, results_text, llm_analysis_text).
    """
    # update globals only for UI/logging; logic uses parameters
    global current_scenario, use_llm_agent
    current_scenario = scenario
    use_llm_agent = bool(llm_agent)

    # Orchestration by hand (simpler than letting LLM pick)
    intent = user_tool(scenario)
    cart = agent_tool(intent, scenario, llm_agent)
    payment = merchant_tool(cart, scenario)
    final = credentials_tool(payment, scenario)

    results = validate_flow(intent, cart, final)
    mandates = {"intent": intent, "cart": cart, "payment": final}

    # Optional: Have the LLM summarize/assess the flow for educational value (short bullet points)
    llm_notes = ""
    if use_llm_agent:
        llm = _get_llm()
        if llm is not None:
            try:
                prompt = (
                    "You are an AP2 payments expert. Summarize the key findings as exactly 3 bullets.\n"
                    "Format strictly as:\n"
                    "- Cause: <root cause in <= 15 words>\n"
                    "- Evidence: <most relevant fact(s) in <= 15 words>\n"
                    "- Outcome: <final decision/status in <= 15 words>\n"
                    "No repetition across bullets. No extra text.\n"
                    f"Mandates: {json.dumps(mandates)}\n"
                    f"Validation: {results}"
                )
                resp = llm.invoke(prompt)
                raw = getattr(resp, "content", "")
                if isinstance(raw, list) and raw:
                    llm_notes = "\n".join([
                        (p.get("text", "") if isinstance(p, dict) else str(p)) for p in raw
                    ])
                else:
                    llm_notes = raw if isinstance(raw, str) else ""
                # Normalize and enforce structure/dedup
                lines = [line.strip() for line in llm_notes.splitlines() if line.strip().startswith("-")]
                # Keep first occurrence for each expected header
                wanted = {"- Cause:": None, "- Evidence:": None, "- Outcome:": None}
                for line in lines:
                    for key in list(wanted.keys()):
                        if wanted[key] is None and line.startswith(key):
                            wanted[key] = line
                # Fallback: if headers missing, take first unique 3 bullets
                bullets = [v for v in wanted.values() if v]
                if len(bullets) < 3:
                    seen = set()
                    for line in lines:
                        low = line.lower()
                        if low not in seen:
                            bullets.append(line)
                            seen.add(low)
                        if len(bullets) == 3:
                            break
                llm_notes = "\n".join(bullets[:3])
            except Exception as e:
                logger.warning("LLM analysis failed: %s", e)
                llm_notes = "(LLM analysis unavailable)"
        else:
            llm_notes = "(LLM disabled or not configured)"

    return mandates, "\n".join(results), llm_notes

# --- Gradio UI ---
demo = gr.Interface(
    fn=run_simulation,
    inputs=[
        gr.Dropdown(
            ["Normal", "Bad Agent", "Bad Merchant", "Expired Mandate", "Invalid Signature"],
            label="Scenario",
        ),
        gr.Checkbox(label="Use LLM Agent (OpenAI)", value=False),
    ],
    outputs=[
        gr.JSON(label="Mandates"),
        gr.Textbox(label="Validation Results", lines=10),
        gr.Textbox(label="LLM Analysis", lines=6),
    ],
    title="AP2 Fraud/Discrepancy Simulator",
    description=(
        "Simulates AP2-style flows with HMAC-signed mandates. Optional LLM agent uses OpenAI via LangChain."
    ),
    flagging_mode="never",
)

# Expose under the conventional name used by Hugging Face Spaces
# Some environments auto-discover `app` rather than `demo`.
demo = demo.queue(default_concurrency_limit=2)
app = demo

if __name__ == "__main__":
    # Launch the Gradio server. On Hugging Face Spaces (SDK=Gradio), this is the documented pattern.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_api=False,
    )
