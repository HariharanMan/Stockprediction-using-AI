import openai

# Initialize the OpenAI client with CoralFlow API
client = openai.Client(
    api_key='cf-hari-7cec3a5c352afea14a4fb8f6ab44d59c77768116bfdd8e61a481f2d9',
    base_url='http://api.coralflow.co/v1'
)

# Create a chat completion
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a chatbot assistant."},
        {"role": "user", "content": "who is the IPL winner in 2024"}
    ]
)

# Print the entire response to understand its structure
print(completion)



# from openai import OpenAI
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-4o-mini-2024-07-18",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)

