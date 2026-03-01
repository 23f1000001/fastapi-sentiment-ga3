from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError
import os
from typing import Annotated

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyzes customer comments using OpenAI structured outputs",
    version="1.0.0",
)

# CORS (keep if your frontend or autograder needs it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────
# Dependency to extract OpenAI key from request
# ────────────────────────────────────────────────

authorization_header = APIKeyHeader(name="Authorization", auto_error=False)
x_openai_key_header = APIKeyHeader(name="X-OpenAI-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def get_openai_api_key(
    request: Request,
    auth: Annotated[str | None, Depends(authorization_header)],
    x_key: Annotated[str | None, Depends(x_openai_key_header)],
    query_key: Annotated[str | None, Depends(api_key_query)],
) -> str:
    # 1. Authorization: Bearer sk-...
    if auth and auth.startswith("Bearer "):
        return auth.removeprefix("Bearer ").strip()

    # 2. Custom header
    if x_key:
        return x_key.strip()

    # 3. Query parameter ?api_key=...
    if query_key:
        return query_key.strip()

    # 4. Try to read from JSON body (some autograders do this)
    try:
        body = await request.json()
        if isinstance(body, dict) and "api_key" in body and isinstance(body["api_key"], str):
            return body["api_key"].strip()
    except Exception:
        pass

    # 5. Fallback to environment variable (your local testing)
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No valid OpenAI API key provided (Authorization header, X-OpenAI-Key, query param, body, or env var)",
    )


# ────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1, max_length=4000)


class SentimentResponse(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    rating: int = Field(..., ge=1, le=5)


# ────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(
    request: Request,
    data: CommentRequest,
    api_key: Annotated[str, Depends(get_openai_api_key)],
):
    if not data.comment.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Comment cannot be empty or only whitespace"
        )

    client = OpenAI(api_key=api_key)

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an objective sentiment analyzer. "
                        "Classify the sentiment of the comment as 'positive', 'negative', or 'neutral'. "
                        "Assign a rating from 1 to 5:\n"
                        "5 = strongly positive\n"
                        "4 = positive\n"
                        "3 = neutral\n"
                        "2 = negative\n"
                        "1 = strongly negative\n\n"
                        "Return ONLY the structured JSON, nothing else."
                    ),
                },
                {"role": "user", "content": data.comment},
            ],
            response_format=SentimentResponse,
            temperature=0.1,
            max_tokens=100,
        )

        parsed = completion.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Structured output parsing failed")

        return parsed

    except OpenAIError as oe:
        error_detail = str(oe)
        if "insufficient_quota" in error_detail or "invalid" in error_detail.lower():
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="OpenAI API key issue (quota exceeded or invalid)"
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI service error: {error_detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)