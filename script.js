//-----------------------------------
// ELEMENT REFERENCES
//-----------------------------------
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const predictBtn = document.getElementById("predictBtn");
const resultDiv = document.getElementById("result");

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const searchResult = document.getElementById("searchResult");

let selectedFile = null;

//-----------------------------------
// CELEBRITY DATASET
//-----------------------------------
const datasetCelebrities = [
"Adriana Lima","Alex Lawther","Alexandra Daddario","Alvaro Morte","alycia dabnem carey",
"Amanda Crew","amber heard","Andy Samberg","Anne Hathaway","Anthony Mackie","Avril Lavigne",
"barack obama","barbara palvin","Ben Affleck","Bill Gates","Bobby Morley","Brenton Thwaites",
"Brian J. Smith","Brie Larson","camila mendes","Chris Evans","Chris Hemsworth","Chris Pratt",
"Christian Bale","Cristiano Ronaldo","Danielle Panabaker","Dominic Purcell","Dwayne Johnson",
"Eliza Taylor","Elizabeth Lail","elizabeth olsen","ellen page","elon musk","Emilia Clarke",
"Emma Stone","Emma Watson","gal gadot","grant gustin","Gwyneth Paltrow","Henry Cavil",
"Hugh Jackman","Inbar Lavi","Irina Shayk","Jake Mcdorman","Jason Momoa","jeff bezos",
"Jennifer Lawrence","Jeremy Renner","Jessica Barden","Jimmy Fallon","Johnny Depp","Josh Radnor",
"Katharine Mcphee","Katherine Langford","Keanu Reeves","kiernen shipka","Krysten Ritter",
"Leonardo DiCaprio","Lili Reinhart","Lindsey Morgan","Lionel Messi","Logan Lerman",
"Madelaine Petsch","Maisie Williams","margot robbie","Maria Pedraza","melissa fumero",
"Marie Avgeropoulos","Miley Cyrus","Mark Ruffalo","Mark Zuckerberg","Millie Bobby Brown",
"Morena Baccarin","Megan Fox","Morgan Freeman","Nadia Hilker","Natalie Dormer",
"Natalie Portman","Neil Patrick Harris","Pedro Alonso","Penn Badgley","Rami Malek",
"Rebecca Ferguson","Richard Harmon","Rihanna","Robert De Niro","Robert Downey Jr",
"Sarah Wayne Callies","scarlett johansson","Selena Gomez","Shakira Isabel Mebarak",
"Sophie Turner","Stephen Amell","Taylor Swift","Tom Cruise","tom ellis","Tom Hardy",
"Tom Hiddleston","Tom Holland","Tuppence Middleton","Ursula Corbero",
"Wentworth Miller","Zac Efron","Zendaya","Zoe Saldana"
];

//-----------------------------------
// SMART SEARCH FUNCTION
//-----------------------------------
searchBtn.onclick = () => {
    const query = searchInput.value.trim().toLowerCase();

    if (query === "") {
        searchResult.innerHTML = "<p style='color:red;'>Please type a name.</p>";
        return;
    }

    // Full-word matching: First name, last name, or full name
    const results = datasetCelebrities.filter(name => {
        const lowered = name.toLowerCase();
        const words = lowered.split(" "); // split full name into words

        // full name match
        if (lowered === query) return true;

        // first name match
        if (words[0] === query) return true;

        // last name match (word[1] may be undefined)
        if (words[1] && words[1] === query) return true;

        return false;
    });

    if (results.length > 0) {
        searchResult.innerHTML = `
            <p style="color:green;">Matches found:</p>
            <ul>${results.map(n => `<li>${n}</li>`).join("")}</ul>
        `;
    } else {
        searchResult.innerHTML = `<p style="color:red;">No match found in dataset.</p>`;
    }
};

//-----------------------------------
// UPLOAD IMAGE LOGIC
//-----------------------------------
uploadBtn.onclick = () => imageInput.click();

imageInput.onchange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    selectedFile = file;
    fileName.textContent = `Selected: ${file.name}`;

    const reader = new FileReader();
    reader.onload = () => {
        imagePreview.innerHTML = `<img src="${reader.result}" />`;
    };
    reader.readAsDataURL(file);
};

//-----------------------------------
// CLEAR BUTTON
//-----------------------------------
clearBtn.onclick = () => {
    selectedFile = null;
    imagePreview.innerHTML = "No Image";
    fileName.textContent = "No file selected";
    resultDiv.innerHTML = "";
    imageInput.value = "";
};

//-----------------------------------
// PREDICTION REQUEST TO BACKEND
//-----------------------------------
predictBtn.onclick = async () => {
    if (!selectedFile) {
        resultDiv.innerHTML = "<p style='color: red;'>Please upload an image first.</p>";
        return;
    }

    resultDiv.innerHTML = "<p>Processing...</p>";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (data.success && data.faces && data.faces.length > 0) {
            let resultHTML = `<h3>Detected ${data.num_faces_detected} face(s):</h3>`;

            data.faces.forEach((face) => {
                resultHTML += `
                <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <h4>Face ${face.face_index + 1}</h4>
                    <p><strong>Celebrity:</strong> ${face.prediction.celebrity_name}</p>
                    <p><strong>Confidence:</strong> ${(face.prediction.confidence * 100).toFixed(2)}%</p>

                    <details>
                        <summary>Top 3 Predictions</summary>
                        <ul>
                            ${face.top_3_predictions
                                .map(
                                    p =>
                                        `<li>${p.celebrity_name} (${(p.confidence * 100).toFixed(2)}%)</li>`
                                )
                                .join("")}
                        </ul>
                    </details>
                </div>
                `;
            });

            resultDiv.innerHTML = resultHTML;

        } else {
            resultDiv.innerHTML =
                "<p style='color:red;'>No face detected or unexpected server response.</p>";
        }

    } catch (err) {
        resultDiv.innerHTML = `<p style='color:red;'>Backend error: ${err.message}</p>`;
    }
};
