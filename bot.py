import websocket, json, pprint

SOCKET = "wss://stream.binance.com:9443/ws/ethusdt@kline_1m"

closes = []


def on_open(ws):
    print("Open Conection!")


def on_close(ws):
    print('Conection closed')


def on_message(ws, msg):
    # print(msg)
    json_msg = json.loads(msg)
    # pprint.pprint(json_msg)

    candle = json_msg['k']
    is_candle_close = candle['x']
    close = candle['c']

    if is_candle_close:
        print("El precio cerró a: {}".format(close))
        closes.append(float(close))


ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()
