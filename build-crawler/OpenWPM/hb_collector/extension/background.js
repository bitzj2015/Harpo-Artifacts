//get the message about the ads from the communicator (content script)
//here will be the logic of the extension.
var url_onupdate;
var ads;
var demandPartners = {}
var store = {};
var num_of_ads = 0;

var start_time;
var end_time;
chrome.tabs.onUpdated.addListener( function(tabId, changeInfo ,tab) {
    url_onupdate = tab.url;
    if(changeInfo.status === 'loading') {
        demandPartners = {};
        start_time = Date.now();
    }
    else if(changeInfo.status === 'complete') {
        //var domainAndPartner = {};
        end_time = Date.now();
        console.log("TOTAL PAGE TIME:" + (end_time - start_time));
        var total_time = end_time - start_time;
        //domainAndPartner[url_onupdate] = demandPartners;
        demandPartners['website'] = url_onupdate;
        demandPartners['pageLoadTime'] = total_time;
        // console.log(demandPartners);
        console.log("url_onupdate" + url_onupdate);
        var tosend = JSON.stringify(demandPartners)
        //demandPartners = {};
        $.ajax({
            url: 'http://127.0.0.1:5050/info',
            type: 'POST',
            data: tosend,
            success: function(res) {
                console.log('success');
            }
        });
    } 
});



function getBidderFromURL(url) {
    console.log("getBidderFromURL listener")

    if (!url) return;
    for (var p in patternsToBidders) {
        var urlRegex;
        var wildcard = p.indexOf('*');
        if (wildcard !== -1) {
            urlRegex = new RegExp(p.slice(0, wildcard) + '.' + p.slice(wildcard));
        }
        if (url.indexOf(p) != -1 || (urlRegex && urlRegex.test(url))) {
            return patternsToBidders[p];
        }
    }
}

var patternsToBidders = {};
var reqPatterns = [];
for (var b in bidderPatterns) {
    var patterns = bidderPatterns[b];
    for (var i = 0; i < patterns.length; i++) {
        var p = patterns[i];
        patternsToBidders[p] = b;
        if (p[0] === '.') p = '*' + p;
        var urlPattern = '*://' + p + '*';
        reqPatterns.push(urlPattern);
    }
}

chrome.webRequest.onBeforeRequest.addListener( function(info) {
        console.log("dp info listener")
    demandPartners[info.requestId] = {"name":getBidderFromURL(info.url), "start_time":info.timeStamp}
}, { 
    urls: reqPatterns
},
    ['requestBody']
);

chrome.runtime.onMessage.addListener(function(request,sender){
    //console.log(request);
    ads = request;
    console.log("request listener")
    console.log(ads.ads);
    var tosend = JSON.stringify(ads.ads);
    console.log(tosend);
    if (ads.type == "pbjs"){
            console.log('sending pbs');

            $.ajax({
                url: 'http://127.0.0.1:5050/pbjs',
                type: 'POST',
                data: tosend,
                success: function(res) {
                    console.log('success');
                }
            });
        }
    else {
        console.log('sending gpt');

        $.ajax({
            url: 'http://127.0.0.1:5050/gpt',
            type: 'POST',
            data: tosend,
            success: function(res) {
                console.log('success');
            }
        });

    }

});

chrome.webRequest.onCompleted.addListener(function(info) {
    console.log(info)
    console.log("dp listener")

    demandPartners[info.requestId]["end_time"] = info.timeStamp;
    demandPartners[info.requestId]["latency"] = info.timeStamp - demandPartners[info.requestId]["start_time"];
}, {
    urls: reqPatterns
},
    ['responseHeaders']
);
