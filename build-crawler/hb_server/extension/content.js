console.log("CONTENT SCRIPT CALLED");
var  flattenObject = (obj) => {
  const flattened = {}

  Object.keys(obj).forEach((key) => {flattened[key] = obj[key]})

  return flattened
}
window.googletag = window.googletag || {};
googletag.cmd = googletag.cmd || [];
window.pbjs = window.pbjs || {};
pbjs.que = pbjs.que || [];
//console.log(pbjs.que.length);
var holder = {
    biddings: {}
};
var ads = {};
gptAuctions = [];

function getBids(){
    var domain = document.domain;
    var n = pbjs.getBidResponses();
    Object.keys(n)
        .forEach(function(e) {
            n[e].bids.forEach(function(n) {

                var t = {
                    ad:[n.ad],
                    domain: domain,
                    adId:n.adId,
                    adUnit:n.adUnit,
                    adUnitCode:n.adUnitCode,
                    auctionId:n.auctionId,
                    bidder:n.bidder,
                    cpm:n.cpm,
                    creativeId:n.creativeId,
                    currency:n.currency,
                    dealId:n.dealId,
                    mediaType:n.mediaType,
                    netRevenue:n.netRevenue,
                    pbAg:n.pbAg,
                    pbCg:n.pbCg,
                    pbDg:n.pbDg,
                    pbHg:n.pbHg,
                    pbLg:n.pbLg,
                    pbMg:n.pbMg,
                    requestId:n.requestId,
                    requestTimestamp:n.requestTimestamp,
                    responseTimestamp:n.responseTimestamp,
                    requestId:n.requestId,
                    height:n.height,
                    width:n.width,
                    size:n.size,
                    source:n.source,
                    statusMessage:n.statusMessage,
                    ttl:n.ttl,
                    timeToRespond:n.timeToRespond, 
                    type:"pbjs"


                };
                console.log(t)
                holder.biddings[e] ? holder.biddings[e].push(t) : holder.biddings[e] = [t]

            })
        })
    holder['domain'] = domain;
    holder['type'] = 'pbjs';

    
    postMessage(holder, "*");
}


function getWinners(){
    var winners = pbjs.getAllWinningBids();
    winners.forEach(function(n){
        holder.biddings[n.adUnitCode].forEach(function(e){
            e.winner = e.bidder === n.bidder && e.cpm === n.cpm
        })
    })
    postMessage(holder,"*");
}


pbjs.que.push(function() {
    pbjs.onEvent("auctionEnd",getBids);
    pbjs.onEvent("bidWon",getWinners);
});
function gptSlotRender(ev) {
    console.log(ev);
    var domain = document.domain;
    var adUnitCode = ev.slot.getSlotElementId();
    gptAuctions[adUnitCode] = gptAuctions[adUnitCode] || [];
    

  
    

    var auctionInfo = {
        campaingId: ev.campaingId,
        adId: ev.advertiserId,
        size: ev.size ? ev.size[0]+'x'+ev.size[1]: "",
        adUnitCode: adUnitCode,
        adUnitPath: ev.slot.getAdUnitPath(),
        // targeting: ev.slot.getTargetingKeys().map(auctionInfo => [auctionInfo, ev.slot.getTargeting(auctionInfo)]),
        targeting: ev.slot.getTargetingKeys(),

        //size: size,
        event: "slotRenderEnded",
        domain: domain,
        time: Date.now(),
        type: "gpt"
    };
    //console.log(ev)
    gptAuctions[adUnitCode].push(auctionInfo);
    
    postMessage(auctionInfo,"*");
    
}


googletag.cmd.push( function() {
    //googletag.pubads().addEventListener('slotRequested', gptSlotRender );
    googletag.pubads().addEventListener('slotRenderEnded', gptSlotRender );
})

