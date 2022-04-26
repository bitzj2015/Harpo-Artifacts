const messageDuration = 20000; // in ms
const interestSegmentUrl = "https://registry.bluekai.com/get_categories";

var categoriesForm = document.getElementById("categoryForm");

var updateMessage = document.getElementById("updateMessage");

var closeButton = document.getElementsByClassName("delete")[0];

var reconnectMessage = document.getElementById("reconnectMessage");

var reconnectButton = document.getElementById("reconnectButton");

var digitalFootprintButton = document.getElementById("digitalFootprintButton");

var categoryButton = document.getElementById("categoryButton");

var categoryPane = document.getElementById("categoryPane");

var digitalFootprintPane = document.getElementById("digitalFootprintPane");

var interestSegmentTableBody = document.getElementById("interestSegmentTableBody");

var previousCategoryState = {};

var reconnectMessageDisplayed = false;

var interestSegmentsLoaded = 0;

digitalFootprintButton.addEventListener('click', () => {
    const HttpRequest = new XMLHttpRequest();
    HttpRequest.open("GET", interestSegmentUrl);
    HttpRequest.send();
    HttpRequest.onload = (e) => {
        var parsedData = JSON.parse(HttpRequest.responseText);

        if (parsedData != null && !interestSegmentsLoaded) {
            console.log(parsedData);
            loadInterestSegments(parsedData);
            interestSegmentsLoaded++;
        } else if (parsedData == null) {
            loadInterestSegments(["No interest segments available"]);
        }
        
        changePanes();
    }

    HttpRequest.onerror = (e) => {
        console.log("A network error occurred when requesting interest segments.");
        loadInterestSegments(["An error occurred when attempting to load interest segments."]);

        changePanes();
    }
})

categoryButton.addEventListener('click', () => {
    hideElement(digitalFootprintPane);
    hideElement(categoryButton);
    showElement(categoryPane);
    showElement(digitalFootprintButton);

    interestSegmentTableBody.innerHTML = "";
    interestSegmentsLoaded = 0;
})

closeButton.addEventListener('click', () => {
    hideElement(updateMessage);
})

reconnectButton.addEventListener('click', () => {
    browser.runtime.sendMessage({'type':'reconnect_request'});

    hideElement(reconnectMessage);
    reconnectMessageDisplayed = false;
})

function hideElement(elem) {
    elem.style.display = "none";
}

function showElement(elem) {
    elem.style.display = "block";
}

document.getElementById("updateButton").addEventListener("click", function() {
    var checkboxes = document.getElementsByClassName("categoryCheckbox");

    var updatedCategories = [];

    for (let i=0; i < checkboxes.length; i++) {
        if (checkboxes[i].checked != previousCategoryState[i].checked) {
            previousCategoryState[i].checked = !(previousCategoryState[i].checked);
            updatedCategories.push(checkboxes[i].id);
        }
    }

    showElement(updateMessage);

    setTimeout(() => {
        hideElement(updateMessage);
    }, messageDuration);

    window.scrollTo(0, 0);

    // code to post changes to background.js and then to rl_agent, rl_agent should save changes to a text file
    browser.runtime.sendMessage({type: 'category_update', updated_categories: updatedCategories});
})

var stringToHTML = function (str) {
	var parser = new DOMParser();
	var doc = parser.parseFromString(str, 'text/html');
	return doc.body;
};

browser.runtime.sendMessage({
    "type": "category_request",
});

browser.runtime.onMessage.addListener(function(message, sender) {
    if (message.type == "category_reply") {
        previousCategoryState = message.categories;
        loadCategoryData(message.categories);
    }

    if (message.type == "connection_failure") {
        if (!reconnectMessageDisplayed) {
            showElement(reconnectMessage);
            reconnectMessageDisplayed = true;
        }
    }
})

function loadCategoryData(categories) {
    for (let key in categories) {
        var checkboxElement = `<li>
                                    <label class="checkbox">
                                        <input class="categoryCheckbox" type="checkbox" id=${key} ${categories[key].checked ? "checked" : ""}>
                                        ${categories[key].name}
                                    </label>
                                </li>`;

        var checkbox = stringToHTML(checkboxElement);
        categoriesForm.appendChild(checkbox);
    }
}

function loadInterestSegments(interestSegments) {
    for (const segment of interestSegments) {
        var segmentInnerHTML = `<tr>
                                    <td>
                                        ${segment}
                                    </td>
                                </tr>`;
        var newSegmentElement = document.createElement("tr");
        newSegmentElement.innerHTML = segmentInnerHTML;
        interestSegmentTableBody.appendChild(newSegmentElement);
    }
}

function changePanes() {
    hideElement(categoryPane);
    hideElement(digitalFootprintButton);
    showElement(digitalFootprintPane);
    showElement(categoryButton);
}

function showLoadingPage() {
    
}