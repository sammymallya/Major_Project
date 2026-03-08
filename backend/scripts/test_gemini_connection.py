"""
Lightweight connectivity test for the Gemini API.

Checks that `GEMINI_API_KEY` from `backend/.env` is present, attempts to list
models (if supported by the installed client), verifies the configured model
(`GEMINI_MODEL`), and performs a minimal generation probe.

Run from project root:
    python -m backend.scripts.test_gemini_connection
"""

from __future__ import annotations

import os
import sys
import logging

from dotenv import load_dotenv

import google.generativeai as genai


# Configure logging to be concise
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    # Load backend/.env only
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    logger.info("Loading environment from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=False)

    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL") or os.environ.get("QUERY_STRUCTURER_MODEL_NAME")

    if not api_key:
        logger.error("GEMINI_API_KEY not found in backend/.env")
        return 1

    if not model:
        logger.warning("GEMINI_MODEL not set; using default model name fallback")
        model = "gemini-2.5-flash"

    logger.info("Configuring Gemini client and probing model=%s", model)
    genai.configure(api_key=api_key)

    # Try to list available models (if client provides that helper)
    try:
        if hasattr(genai, "list_models"):
            lm = genai.list_models()
            # Try several ways to introspect returned value
            model_names = set()
            try:
                # Some clients return an object with .models
                for m in getattr(lm, "models", []) or []:
                    # m may be dict-like or object
                    if isinstance(m, dict):
                        name = m.get("name") or m.get("model")
                    else:
                        name = getattr(m, "name", None) or getattr(m, "model", None)
                    if name:
                        model_names.add(name)
            except Exception:
                pass
            try:
                if isinstance(lm, dict):
                    model_names.update(lm.keys())
            except Exception:
                pass

            logger.info("list_models returned %d candidate names", len(model_names))
            model_available = model in model_names if model_names else None
            logger.info("Model %s available: %s", model, model_available)
        else:
            logger.info("Client has no list_models helper; skipping model listing")
            model_available = None
    except Exception as e:
        logger.warning("list_models call failed: %s", e)
        model_available = None

    # As a final check, attempt a tiny generation to confirm the model responds
    probe_prompt = "Respond with: PONG"
    try:
        # Try generate() if present
        if hasattr(genai, "generate"):
            resp = genai.generate(model=model, input=probe_prompt, temperature=0.0)
            # Attempt to extract string result
            out = None
            if isinstance(resp, str):
                out = resp
            else:
                try:
                    out = getattr(resp, "text", None) or getattr(resp, "content", None)
                except Exception:
                    out = str(resp)
            logger.info("Generation probe succeeded (truncated): %s", (out or "").strip()[:200])
            return 0

        # Try chat.create
        if hasattr(genai, "chat") and hasattr(genai.chat, "create"):
            messages = [{"role": "system", "content": "You are a test agent."}, {"role": "user", "content": probe_prompt}]
            resp = genai.chat.create(model=model, messages=messages, temperature=0.0)
            out = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
            logger.info("Chat probe succeeded (truncated): %s", (out or "").strip()[:200])
            return 0

        # Try GenerativeModel wrapper
        try:
            gm = genai.GenerativeModel(model, system_instruction="You are a test agent.")
            resp = gm.generate_content(probe_prompt, generation_config={"temperature": 0.0})
            out = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
            logger.info("GenerativeModel probe succeeded (truncated): %s", (out or "").strip()[:200])
            return 0
        except Exception as e:
            logger.warning("GenerativeModel probe failed: %s", e)

        logger.error("No supported generation method available on installed genai client")
        return 2

    except Exception as e:
        logger.error("Generation probe failed: %s", e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
