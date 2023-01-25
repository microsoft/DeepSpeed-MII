import mii

print(f"Querying flan...")

str = "Tell me why DeepSpeed is the greatest."

generator = mii.mii_query_handle("flan_deployment")
result = generator.query({
    'query': str,
    'conversation_id': 1,
    'past_user_inputs': [],
    'generated_responses': []
})

print(result)
print(f"time_taken: {result.time_taken}")
