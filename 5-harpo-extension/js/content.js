console.log("content script successfully loaded for url");

if (document.readyState !== 'loading') {
    console.log('document is already ready, just execute code here');
    onDocumentLoad();
} else {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('document was not ready, place code here');
        onDocumentLoad();
    });
}

function onDocumentLoad() {
    console.log('in send function');

    browser.runtime.sendMessage({
        "type": "send_page_info",
        "url": window.location.href,
        "html": document.documentElement.innerHTML
    });

    browser.runtime.sendMessage({
        "type": "page_creation"
    })
}