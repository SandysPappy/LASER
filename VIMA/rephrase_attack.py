from openai import OpenAI

client = OpenAI(api_key="")

def rephrase_attack(rephrasing_prefix, curr_prompt):
    response = client.completions.create(model="gpt-3.5-turbo-instruct",
    prompt=rephrasing_prefix + curr_prompt,
    temperature=0.99,
    max_tokens=512,
    n=1,
    stop=".")
    result = response.choices[0].text.replace("\n", "")
    return result