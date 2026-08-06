from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from core.config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MAX_OUTPUT_TOKENS,
    GROQ_MAX_RETRIES,
    GROQ_MIN_REQUEST_INTERVAL_SECONDS,
    GROQ_MODEL,
    LLM_PROVIDER,
)


GROQ_RATE_LIMITER = InMemoryRateLimiter(
    requests_per_second=1 / GROQ_MIN_REQUEST_INTERVAL_SECONDS,
    check_every_n_seconds=0.5,
    max_bucket_size=1,
)


def make_llm():
    if LLM_PROVIDER == "groq":
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=GROQ_MAX_OUTPUT_TOKENS,
            max_retries=GROQ_MAX_RETRIES,
            rate_limiter=GROQ_RATE_LIMITER,
        )
    return ChatGoogleGenerativeAI(
        api_key=GOOGLE_API_KEY,
        model=GEMINI_MODEL,
        temperature=0,
        max_output_tokens=GROQ_MAX_OUTPUT_TOKENS,
    )