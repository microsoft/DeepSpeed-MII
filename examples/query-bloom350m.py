import mii

generator = mii.mii_query_handle("gpt2_deployment")
result = generator.query({'query': "DeepSpeed is"}, do_sample=True)
