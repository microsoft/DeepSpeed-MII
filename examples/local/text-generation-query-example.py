import mii
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--deployment',
                    '-d',
                    type=str,
                    required=True,
                    help="deployment_name set in the MII deployment")
args = parser.parse_args()

generator = mii.mii_query_handle(args.deployment)
result = generator.query({'query': ["DeepSpeed is the", "Seattle is in"]}) #, max_new_tokens=50, do_sample=True, top_p=0.9)
print(result.response)
print("time_taken:", result.time_taken)
