// Create WebSocket connection.
// const WebSocket = require('ws');
var socket = new WebSocket('ws://host.docker.internal:8765');

// Connection opened
socket.addEventListener('open', function (event) {
    var myObj = {"html":"this is a dummy page", "url":"www.dummy.com", "type": "url_request"};
    console.log("message sent:", myObj)
    socket.send(JSON.stringify(myObj));
});


// Listen for messages

socket.addEventListener('message', function (event) {
    console.log('Message from server ', event.data);
});
