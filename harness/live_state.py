"""Holds the engine loaded by harness.live. Never reloaded, so the engine survives code hot-reloads."""
LLM = None
DESC = ""
MODE = ""  # harness.session.ENGINE_MODES
