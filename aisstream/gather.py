import asyncio
import json
from collections import defaultdict
import os
from pathlib import Path
import datetime

import websockets

"""
UserID below is MMSI
"""

CAPTURE_DURATION = datetime.timedelta(hours=3, minutes=0, seconds=0)
DATA_FOLDER = Path("data-unfiltered")
BOUNDING_BOXES = [[57.45, 11.14], [57.71, 11.89]]
MESSAGE_FILTER = ["PositionReport", "ShipStaticData"]

with open(".api_key", "r") as f:
    API_KEY = f.readline()

# both map UserID -> Value
position_data = defaultdict(list)
ship_data = {}


async def connect_ais_stream():
    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        subscribe_message = {
            "APIKey": API_KEY,
            "BoundingBoxes": [BOUNDING_BOXES],
            "FilterMessageTypes": MESSAGE_FILTER,
        }
        subscribe_message_json = json.dumps(subscribe_message)
        await websocket.send(subscribe_message_json)

        message_counter = 1
        start_time = datetime.datetime.now()
        async for message_json in websocket:
            time_passed = datetime.datetime.now() - start_time
            print(
                f"\rElapsed time: {time_passed}. Messages received: {message_counter}",
                end="",
            )
            message_counter += 1

            if time_passed > CAPTURE_DURATION:
                break

            message = json.loads(message_json)
            metadata = message["MetaData"]
            message_type = message["MessageType"]

            match message_type:
                case "PositionReport":
                    position_report = message["Message"]["PositionReport"]
                    position_report["time_utc"] = metadata["time_utc"]
                    user_id = position_report["UserID"]
                    position_data[user_id].append(position_report)

                case "ShipStaticData":
                    ship_static_data = message["Message"]["ShipStaticData"]
                    user_id = ship_static_data["UserID"]
                    ship_data[user_id] = ship_static_data


if __name__ == "__main__":
    try:
        asyncio.run(connect_ais_stream())
    except Exception as e:
        print()
        print(e)
    except KeyboardInterrupt:
        pass

    print()
    print(f"Number of position keys: {len(position_data)}")
    print(f"Number of ship keys: {len(ship_data)}")

    os.makedirs(DATA_FOLDER, exist_ok=True)
    num_collected = len(os.listdir(DATA_FOLDER)) // 2

    capture_id = num_collected + 1

    with open(DATA_FOLDER / f"position-capture-{capture_id}.json", "w") as f:
        json.dump(position_data, f)
    with open(DATA_FOLDER / f"ship-info-capture-{capture_id}.json", "w") as f:
        json.dump(ship_data, f)
