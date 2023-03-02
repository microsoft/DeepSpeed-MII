import time
import threading
import mii
from flask import Flask, request
from flask_restful import Resource, Api
from werkzeug.serving import make_server
from mii.constants import SERVER_SHUTDOWN_TIMEOUT, RESTFUL_API_PATH
from google.protobuf.json_format import MessageToJson


def shutdown(thread):
    time.sleep(SERVER_SHUTDOWN_TIMEOUT)
    thread.server.shutdown()


def createRestfulGatewayApp(task, mii_config, server_thread):
    # client must be thread-safe
    client = mii.MIIClient(task, "localhost", mii_config.port_number)

    class RestfulGatewayService(Resource):
        def __init__(self):
            super().__init__()

        def post(self):
            data = request.get_json()
            kwargs = data["kwargs"] if "kwargs" in data else {}
            result = client.query(data["request"], **kwargs)
            return MessageToJson(result)

    app = Flask("RestfulGateway")

    @app.route("/terminate", methods=['GET'])
    def terminate():
        # Need to shutdown *after* completing the request
        threading.Thread(target=shutdown, args=(server_thread, )).start()
        return "Shutting down RESTful API gateway server"

    api = Api(app)
    path = '/{}/{}'.format(RESTFUL_API_PATH, task)
    api.add_resource(RestfulGatewayService, path)

    return app


class RestfulGatewayThread(threading.Thread):
    def __init__(self, task, mii_config):
        threading.Thread.__init__(self)
        self.mii_config = mii_config

        app = createRestfulGatewayApp(task, mii_config, self)
        self.server = make_server('127.0.0.1', mii_config.restful_api_port, app)
        self.ctx = app.app_context()
        self.ctx.push()

        self._stop_event = threading.Event()

    def run(self):
        self.server.serve_forever()

    def get_stop_event(self):
        return self._stop_event
