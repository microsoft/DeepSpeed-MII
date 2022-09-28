import mii
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--deployment',
                    '-d',
                    type=str,
                    required=True,
                    help="deployment_name set in the MII deployment")
args = parser.parse_args()

rank = 0
generator = mii.mii_query_handle(f"gptj_{rank}")
result = generator.query({'query': ["DeepSpeed is the", "Seattle is"]})
print(result.response)
print("time_taken:", result.time_taken)

rank = 1
generator2 = mii.mii_query_handle(f"gptj_{rank}")
result2 = generator2.query({'query': ["DeepSpeed is the", "Seattle is"]})
print(result.response)
print("time_taken:", result2.time_taken)

for i in range(10):
    result = generator.query(
        {'query': ["DeepSpeed is the",
                   "Seattle is",
                   "Rhode Island is",
                   "Vermont is"]})
    result2 = generator2.query(
        {'query': ["DeepSpeed is the",
                   "Seattle is",
                   "Rhode Island is",
                   "Vermont is"]})
    print(f"time taken: {result.time_taken}, {result2.time_taken}")
    print(result, result2)
