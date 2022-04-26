console.log("hello world: " + window.location.href);

function html2text(html) {
    var tag = document.createElement('div');
    tag.innerHTML = html;

    return tag.innerText;
}
var message = html2text(document.documentElement.innerHTML); // what is this?
console.log("orig: " + message.length + window.location.href);
// browser.runtime.sendMessage({"url": window.location.href, "text":message});

browser.runtime.onMessage.addListener(request => {
    console.log("Message from the background script:");
    console.log(request.greeting);
    // var message = html2text(document.documentElement.innerHTML);
    browser.runtime.sendMessage({ "url": window.location.href, "html": document.documentElement.innerHTML });
    console.log("Sending: " + message + window.location.href);
    return Promise.resolve({ response: "Hi from content script" });
});