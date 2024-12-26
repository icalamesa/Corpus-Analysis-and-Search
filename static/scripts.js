const apiBaseUrl = "http://127.0.0.1:5000";

// Load File List
async function loadFileList() {
    const response = await fetch(`${apiBaseUrl}/files`); // Fixed template literal syntax
    if (response.ok) {
        const files = await response.json();
        const fileList = document.getElementById("file-list");
        fileList.innerHTML = ""; // Clear existing rows
        files.forEach((file, index) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${file}</td>
            `;
            fileList.appendChild(row); // Append the row to the table body
        });
    } else {
        console.error("Failed to load file list.");
    }
}

// Search Corpus
async function searchCorpus() {
    const query = document.getElementById("query").value;

    const response = await fetch(`${apiBaseUrl}/search`, { // Fixed template literal syntax
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
    });

    if (response.ok) {
        const data = await response.json();

        // Populate the search results table
        const searchResultsTable = document.getElementById("search-results");
        searchResultsTable.innerHTML = ""; // Clear previous results

        data.forEach((result) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${result.File}</td>
                <td>${result.Similarity.toFixed(2)}</td>
            `;
            searchResultsTable.appendChild(row); // Append the row to the table body
        });
    } else {
        console.error("Failed to search corpus.");
    }
}

// Analyze Corpus
async function analyzeCorpus() {
    const response = await fetch(`${apiBaseUrl}/analyze`); // Fixed template literal syntax
    if (response.ok) {
        const data = await response.json();

        // Populate the analysis results table
        const analyzeResultsTable = document.getElementById("analyze-results");
        analyzeResultsTable.innerHTML = ""; // Clear previous results

        Object.entries(data).forEach(([file, metrics]) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${file}</td>
                <td>${metrics.total_tokens}</td>
                <td>${metrics.unique_tokens}</td>
                <td>${metrics.lexical_diversity.toFixed(2)}</td>
                <td>${metrics.average_token_length.toFixed(2)}</td>
                <td>${metrics.formality_ratio.toFixed(2)}</td>
            `;
            analyzeResultsTable.appendChild(row); // Append the row to the table body
        });
    } else {
        console.error("Failed to analyze corpus.");
    }
}

// Reload Corpus
async function reloadCorpus() {
    const response = await fetch(`${apiBaseUrl}/reload`); // Fixed template literal syntax
    if (response.ok) {
        const data = await response.json();
        document.getElementById("reload-results").textContent = JSON.stringify(data, null, 2);
    } else {
        console.error("Failed to reload corpus.");
    }
}

// Drag and Drop
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

function triggerFileInput() {
    fileInput.click();
}

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.borderColor = "green";
});

dropArea.addEventListener("dragleave", () => {
    dropArea.style.borderColor = "#ccc";
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.style.borderColor = "#ccc";

    const files = e.dataTransfer.files;
    handleFileUpload(files);
});

fileInput.addEventListener("change", (e) => {
    const files = e.target.files;
    handleFileUpload(files);
});

async function handleFileUpload(files) {
    const formData = new FormData();
    for (const file of files) {
        formData.append("files", file);
    }

    const response = await fetch(`${apiBaseUrl}/upload`, { // Fixed template literal syntax
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        const result = await response.json();
        uploadStatus.textContent = `Uploaded successfully:\n${JSON.stringify(result, null, 2)}`;
        loadFileList(); // Refresh file list
    } else {
        uploadStatus.textContent = `Error uploading files: ${response.statusText}`;
    }
}

// Load file list on page load
window.onload = loadFileList;
