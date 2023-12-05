# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import time
import threading

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from werkzeug.serving import make_server

import mii
from mii.constants import RESTFUL_GATEWAY_SHUTDOWN_TIMEOUT, RESTFUL_API_PATH


def shutdown(thread):
    time.sleep(RESTFUL_GATEWAY_SHUTDOWN_TIMEOUT)
    thread.server.shutdown()


def createRestfulGatewayApp(deployment_name, server_thread):
    class RestfulGatewayService(Resource):
        def __init__(self):
            super().__init__()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.client = mii.client(deployment_name)

        def post(self):
            data = request.get_json()
            result = self.client.generate(**data)
            return jsonify([r.to_msg_dict() for r in result])

    app = Flask("RestfulGateway")

    @app.route("/terminate", methods=["GET"])
    def terminate():
        # Need to shutdown *after* completing the request
        threading.Thread(target=shutdown, args=(server_thread, )).start()
        return "Shutting down RESTful API gateway server"

    api = Api(app)
    path = "/{}/{}".format(RESTFUL_API_PATH, deployment_name)
    api.add_resource(RestfulGatewayService, path)

    return app


class RestfulGatewayThread(threading.Thread):
    def __init__(self, deployment_name, rest_host, rest_port, rest_procs):
        threading.Thread.__init__(self)

        app = createRestfulGatewayApp(deployment_name, self)
        self.server = make_server(rest_host,
                                  rest_port,
                                  app,
                                  threaded=False,
                                  processes=rest_procs)
        self.ctx = app.app_context()
        self.ctx.push()

        self._stop_event = threading.Event()

    def run(self):
        self.server.serve_forever()
        self._stop_event.set()

    def get_stop_event(self):
        return self._stop_event
