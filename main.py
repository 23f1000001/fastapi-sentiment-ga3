from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key from environment (recommended)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# If you prefer hard-coded for quick testing (not recommended):
# client = OpenAI(api_key="sk-....your-key....")

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    rating: int = Field(..., ge=1, le=5)


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(data: CommentRequest):
    if not data.comment or not data.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",               # or "gpt-4.1-mini" if that's the real name in 2026
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analyzer. "
                        "Respond ONLY with the structured JSON - no extra text. "
                        "Classify sentiment as 'positive', 'negative' or 'neutral'. "
                        "Give rating 1–5 where 5 = very positive, 1 = very negative."
                    )
                },
                {"role": "user", "content": data.comment}
            ],
            response_format=SentimentResponse,   # ← Pydantic model → auto schema + strict mode
            temperature=0.0,                     # deterministic
            max_tokens=80,
        )

        parsed = completion.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Model did not return parsed structured output")

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")