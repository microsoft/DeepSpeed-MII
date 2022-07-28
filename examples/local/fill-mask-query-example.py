import mii

# roberta
name = "roberta-base"
mask = "<mask>"
# bert
name = "bert-base-uncased"
mask = "[MASK]"
print(f"Querying {name}...")

generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({'query': "Hello I'm a " + mask + " model."})
print(result.response)
print("time_taken:", result.time_taken)
