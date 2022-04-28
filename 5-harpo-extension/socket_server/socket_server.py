#!/usr/bin/python3
import asyncio
import websockets
import app 
import json
import time
import html2text

from category_data import create_category_data_file, write_category_data, read_category_data

host = "0.0.0.0"
port = 8765

category_dict = create_category_data_file()


def category_request_handler(message):
    tab_id = message['id']
    return {"type": "category_reply", "id": tab_id, "categories": category_dict}

def category_update_handler(message):
    
    category_update = message['updated_categories']

    was_updated = False

    for index in category_update:
        int_index = int(index)
        category_dict[index]['checked'] = not category_dict[index]['checked']
        was_updated = True

    if was_updated:
        write_category_data(category_dict, "./category_data.txt")

    print(read_category_data("./category_data.txt"))

    return {"type": "category_update", "status": "successful"}


# set a 'type' attribute for every message_response        
async def handler(websocket, path):
    async for message in websocket:
        try:
            deserialized_message = json.loads(message)
            print("message received", deserialized_message)
        except:
            print("failed to deserialize JSON message from client")

        if deserialized_message['type'] == "send_page_info":
            message_response = app.save_html2text(deserialized_message)
        
        elif deserialized_message['type'] == "url_request":
            try:
                message_response = app.obfuscation_url()
                print("obfuscation url generated: ", message_response)
                message_response['type'] = "url_request"
            except Exception as e:
                print(e)

        elif deserialized_message['type'] == "category_request":
            message_response = category_request_handler(deserialized_message)

        elif deserialized_message['type'] == "category_update":
            message_response = category_update_handler(deserialized_message)
          
        elif deserialized_message['type'] == "test":
            message_response = "response"
            print("test message recieved")

        serialized_message = json.dumps(message_response)

        await websocket.send(serialized_message)


async def main():
    print("Web socket server now running on {} at port {} ...".format(host, port))
    async with websockets.serve(handler, host, port, max_size=None):
        await asyncio.Future()  # run forever

asyncio.run(main())
