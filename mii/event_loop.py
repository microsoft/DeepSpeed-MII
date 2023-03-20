import asyncio
import threading

global event_loop
event_loop = asyncio.get_event_loop()
threading.Thread(target=event_loop.run_forever, daemon=True).start()


def get_event_loop():
    return event_loop
