/*
    All the functions and queues associated with opening
    and maintaining hidden tabs
*/

// queues and related functions
const urlQueue = [];
const hiddenTabs = [];

// constants
const min_tab_duration = 10000; // in ms
const max_tab_duration = 60000; // in ms

/*
    Utilities for hidden tab array
*/
function removeTabRecord(arr, value) {
    var index = arr.indexOf(value);
    if (index > -1) {
      arr.splice(index, 1);
    }
    return arr;
}

// function findArrayItem(arr, val) {
//     var index = arr.findIndex(cur_item => cur_item == val);

//     if (index == -1) {
//         return false;
//     } else {
//         return true;
//     }
// }

// randomly select how long the tab remains open, within specified limits
function genTabDuration(min_duration, max_duration) {
    return Math.ceil(Math.random() * (max_duration - min_duration)) + min_duration;
}

function openHiddenTabs(response) {
    console.log("recieved response and preparing to open hidden tabs");
    function onCreated(tab) {
        try {
            hiddenTabs.push(tab.id); // record hidden tab in list

            browser.tabs.hide(tab.id);
            console.log(`Created and hide new tab: ${tab.id} for obfuscation url.`);
            setTimeout(() => {
                try {
                    browser.tabs.remove(tab.id);
                } catch (error) {
                    console.log(`Error: tried to close hidden tab ${tab.id}, but it does not exist`)
                }
                console.log(`Timeout for new tab: ${tab.id} for obfuscation url.`);
            }, genTabDuration(min_tab_duration, max_tab_duration));

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