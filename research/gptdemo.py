from openai import OpenAI
import os
OPENAI_API_KEY = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "解释量子纠缠"}]
)
print(response.choices[0].message.content)