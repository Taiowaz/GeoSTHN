from openai import OpenAI
import os

os.environ["http_proxy"]="http://10.61.2.90:1082"
os.environ["https_proxy"]="http://10.61.2.90:1082"

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="你是什么"
)

print(response.output_text)