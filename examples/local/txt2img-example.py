import mii
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", action="store_true", help="query")
args = parser.parse_args()

if not args.query:
    mii_configs = {"tensor_parallel": 1, 
                   "dtype": "fp16", 
                   "hf_auth_token": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                   "port_number": 50050}
    mii.deploy(task='text-to-image',
               model="CompVis/stable-diffusion-v1-4",
               deployment_name="sd_deploy",
               mii_config=mii_configs)
else:
    generator = mii.mii_query_handle("sd_deploy")
    result = generator.query({'query': ["a panda in space with a rainbow", "a soda can ontop a snowy mountain"]})
    from PIL import Image
    for idx, img_bytes in enumerate(result.images):
        size = (result.size_w, result.size_h)
        img = Image.frombytes(result.mode, size, img_bytes)
        img.save(f"test-{idx}.png")
