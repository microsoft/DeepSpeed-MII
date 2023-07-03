import mii
import time
generator = mii.mii_query_handle("first_test", "bloom560m_deployment")
result = generator.query({"query": ["DeepSpeed is", "Seattle is"]}, do_sample=True, max_new_tokens = 30, deployment_name = "bloom560m_deployment")
print(result)

time.sleep(5)
generator2 = mii.mii_query_handle("first_test", "microsoft/DialogRPT-human-vs-rand_deployment")
result = generator2.query({'query': "DeepSpeed is the greatest"}, deployment_name = "microsoft/DialogRPT-human-vs-rand_deployment")
print(result)

