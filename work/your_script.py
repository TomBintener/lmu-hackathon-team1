import os

from dotenv import load_dotenv
from google import genai

load_dotenv(".env", override=True)

with genai.Client(
    vertexai=False,
    api_key=os.environ["GEMINI_API_KEY"],
) as client:
    response = client.models.generate_content(
        model=os.environ["GOOGLE_GENAI_MODEL"],
        contents="Reply with exactly OK.",
    )
    print(response.text)
