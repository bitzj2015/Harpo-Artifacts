var s = document.createElement('script');

s.src = chrome.extension.getURL('content.js');
s.onload = function() {
    this.remove();
};

(document.head || document.documentElement).appendChild(s);

function isEmpty(obj) {
    return Object.keys(obj).length === 0;
}

//get the message about the ads from the injected code
window.addEventListener("message", function(e){
    
    if (e.data.biddings && isEmpty(e.data.biddings) === false){
        console.log(e.data)
        chrome.runtime.sendMessage({ads:e.data, type:"pbjs"},function() {});
    } else if (e.data.event === "slotRenderEnded") {
        chrome.runtime.sendMessage({ads:e.data, type:"gpt"},function() {});
    }
})
