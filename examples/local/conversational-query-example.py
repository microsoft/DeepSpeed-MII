import mii

# gpt2
name = "microsoft/DialoGPT-small"

print(f"Querying {name}...")

str = "DeepSpeed is the greatest"

generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({
    'text': str,
    'conversation_id': 3,
    'past_user_inputs': [],
    'generated_responses': []
})

print(result)
print(f"time_taken: {result.time_taken}")

str = "How is DeepSpeed?"
result = generator.query({
    'text': str,
    'conversation_id': result.conversation_id,
    'past_user_inputs': result.past_user_inputs,
    'generated_responses': result.generated_responses
})

print(result)
print("time_taken:", result.time_taken)
