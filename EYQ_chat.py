import os

import time

from dotenv import load_dotenv
from openai import AzureOpenAI
load_dotenv()

client = AzureOpenAI(
    azure_endpoint='https://eyq-incubator.america.fabric.ey.com/eyq/us/api',
    api_key=os.getenv("OPENAI_KEY"),
    api_version='2025-04-01-preview'
)


# global answer

def generateChatResponse(prompt):
    print("Generating LLM response...")
    system_content = f"""
    You are helpful AI Assistant.
    """

    user_content = f"""
        {prompt}
    """

    messages = []
    messages.append({"role": "system", "content": system_content})

    question = {}
    question['role'] = 'user'
    question['content'] = user_content
    messages.append(question)
    report = []
    delay_time_seconds = 0.03

    for resp in client.chat.completions.create(model="gpt-4.1", messages=messages, temperature=0,
                                               n=1,
                                               max_tokens=4000, stream=True, seed=42):

        if resp.choices:
            if resp.choices[0].delta.content:
                report.append(resp.choices[0].delta.content)

                result = "".join(report).strip()
                time.sleep(delay_time_seconds)

    return result


# --- Function 2: Generate Embedding ---
def generateEmbedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    # The embedding is a list of floats
    return response.data[0].embedding

# if __name__ == "__main__":
#     print(generateChatResponse("What is the purpose of teaching ?"))
#     print(generateEmbedding("What is the size of moon ?"))