# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse

from mii import LoadBalancerConfig

from mii.grpc_related.modelresponse_server import serve_load_balancing
from .utils import decode_config_from_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-balancer",
                        type=str,
                        default=None,
                        help="base64 encoded load balancer config")

    args = parser.parse_args()
    assert args.load_balancer is not None, "lb_config required to use load balancer"
    lb_config_dict = decode_config_from_str(args.load_balancer)
    lb_config = LoadBalancerConfig(**lb_config_dict)

    print(f"Starting load balancer on port: {lb_config.port}")
    serve_load_balancing(lb_config)


if __name__ == "__main__":
    # python -m mii.launch.load_balance_server
    main()
