/**************** Define constants and functions ****************/
// Constants
const visitTimeout = 20000;
const ws_url = "wss://localhost:8000";

// function sleep(ms) {
//   return new Promise(resolve => setTimeout(resolve, ms));
// }

// Visit url on the background based on AJAX.
const sendPageInfo = function(url, data, callback) {
    xhr = new XMLHttpRequest();
    try {
        xhr.open('POST', url, true);
        // xhr.withCredentials = true; // send cookies anyway
        xhr.onreadystatechange = function() {
            if (xhr.readyState == XMLHttpRequest.DONE) {
                callback(JSON.parse(xhr.responseText));
            }
        }
        xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        xhr.send(JSON.stringify(data));
        // console.log("Visited obfuscation URL: " + url + " from the background.");
    } catch (e) {
        console.error(e);
    }
}

// Send message from background scripts to content scripts.
function sendMessageToTabs(tabs, url) {
    for (let tab of tabs) {
        // console.log(url + ":" + tab.url, tab.url == url);
        if (tab.url == url) {
            browser.tabs.sendMessage(
                tab.id, { greeting: "Hi from background script" }
            );
            console.log("Send request for html file to content script.");
        }

    }

}

function openHiddenTabs(response) {
    function onCreated(tab) {
        try {
            browser.tabs.hide(tab.id);
            console.log(`Created and hide new tab: ${tab.id} for obfuscation url.`);
            setTimeout(() => {
                browser.tabs.remove(tab.id);
                console.log(`Timeout new tab: ${tab.id} for obfuscation url.`);
            }, 10000);

        } catch (e) {
            console.log(`Created and hide new tab for obfuscation url ${tab.url} failed`, e);
        }
        return true;
    }

    function onError(error) {
        console.log(`Error: ${error}`);
    }

    var i;
    for (i = 0; i < response["obfuscation url"].length; i++) {
        browser.tabs.create({
            active: false,
            url: response["obfuscation url"][i]
        }).then(onCreated, onError);
        // await sleep(500); // why is there some delay needed here?
    }
}

// Receive message from content scripts.
function receiveMessageFromTabs(message) {
    console.log("Received html file from site: " + message.url + ".");
    var num_obfuscation_url = 0;
    while (Math.random() < 0.5) {
        num_obfuscation_url += 1
    }
    // port.postMessage({ "url": message.url, "html": message.html, "num_obfuscation_url": num_obfuscation_url });
    var data = { "url": message.url, "html": message.html, "num_obfuscation_url": num_obfuscation_url };
    sendPageInfo(api_url, data, openHiddenTabs);
    console.log("Sent url and html file to rl_agent.");
}

/**************** Define APIs ****************/
/*
On startup, connect to the "rl_agent" application in user computer.
*/
// var port = browser.runtime.connectNative("rl_agent");

/*
Once the web navigation is completed, send a message to content script to get the html file of active tab.
*/
// browser.webNavigation.onCompleted.addListener(evt => {
//     // Filter out any sub-frame related navigation event
//     if (evt.frameId !== 0) {
//         return;
//     }
//     const url = new URL(evt.url);
//     console.log("Completed the web navigation of site: " + evt.url + ".");
//     browser.tabs.query({
//         currentWindow: true,
//         active: true
//     }).then(tabs => sendMessageToTabs(tabs, url));
// });

/*
Listen the message (url and html) from content scripts, and send it to rl_agent.
*/
browser.runtime.onMessage.addListener(receiveMessageFromTabs);

/*
Listen for the response message from the rl_agent (an obfuscation url), and visit it.
*/
// port.onMessage.addListener((response) => {
//     console.log("Received obfuscation URL: " + response["obfuscation url"] + "," + response["obfuscation url"].length + ".");
//     // sendXhr(response["obfuscation url"]);
//     openHiddenTabs();
// });