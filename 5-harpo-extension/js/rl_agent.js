class RLAgent {
    constructor (websocket_url, reconnect_timeout) {
        this.messageListeners = {};
        this.socket_url = websocket_url;
        this.reconnect_timeout = reconnect_timeout;
    }

    connect() {
        this.socket = new WebSocket(this.socket_url);

        console.log("Connected to RLAgent successfully.");

        this.socket.addEventListener('message', (event) => {
            var parsed_data = JSON.parse(event.data);

            if (parsed_data) {
                var message_type = parsed_data.type;

                this.messageListeners[message_type](parsed_data);
            } else {
                console.log("empty message received from RLAgent")
            }
        });

        this.socket.addEventListener('error', (event) => {
            console.log(`Error connecting to RLAgent, attempting to reconnect...`)
            setTimeout(() => {
                this.connect();
            }, this.reconnect_timeout);
        })
    }
    
    sendPageInfo(url, html) {
        var data = {"type": "send_page_info", "url": url, "html": html};
        this.socket.send(JSON.stringify(data)); // send data to rl_agent
    }

    requestURL() {
        this.socket.send(JSON.stringify({ type : "url_request" }));
    }

    requestURLCategories(tabId) {
        this.socket.send(JSON.stringify({ type : "category_request", id: tabId}));
    }

    sendCategoryUpdate(update) {
        this.socket.send(JSON.stringify(update));
    }

    sendIntSegUpdate(data, uuid) {
        this.socket.send(JSON.stringify({type: "interest_segment_update", uuid: uuid, data: data}));
    }

    addMessageListener(messageType, callback) {
        this.messageListeners[messageType] = callback;
    }
}