// const uploadBtn = document.getElementById("uploadBtn");
// const clearBtn = document.getElementById("clearBtn");
// const imageInput = document.getElementById("imageInput");
// const imagePreview = document.getElementById("imagePreview");
// const fileName = document.getElementById("fileName");
// const predictBtn = document.getElementById("predictBtn");
// const resultDiv = document.getElementById("result");

// let selectedFile = null;

// // Your FastAPI backend URL
// const API_URL = "http://127.0.0.1:8000/predict";

// uploadBtn.onclick = () => imageInput.click();

// imageInput.onchange = (event) => {
//     const file = event.target.files[0];
//     if (!file) return;

//     selectedFile = file;
//     fileName.textContent = `Selected: ${file.name}`;

//     const reader = new FileReader();
//     reader.onload = () => {
//         imagePreview.innerHTML = `<img src="${reader.result}" />`;
//     };
//     reader.readAsDataURL(file);
// };

// clearBtn.onclick = () => {
//     selectedFile = null;
//     imagePreview.innerHTML = "No Image";
//     fileName.textContent = "No file selected";
//     resultDiv.textContent = "";
//     imageInput.value = "";
// };

// predictBtn.onclick = async () => {
//     if (!selectedFile) {
//         resultDiv.textContent = "Please upload an image first.";
//         return;
//     }

//     resultDiv.textContent = "Processing...";

//     const formData = new FormData();
//     formData.append("file", selectedFile);

//     try {
//         const response = await fetch(API_URL, {
//             method: "POST",
//             body: formData,
//         });

//         const data = await response.json();

//         if (data.prediction) {
//             resultDiv.textContent = `Predicted Celebrity: ${data.prediction}`;
//         } else {
//             resultDiv.textContent = "Unexpected response from server.";
//         }
//     } catch (error) {
//         resultDiv.textContent = "Error connecting to backend.";
//     }
// };

const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const predictBtn = document.getElementById("predictBtn");
const resultDiv = document.getElementById("result");

let selectedFile = null;

// Your FastAPI backend URL
const API_URL = "http://127.0.0.1:8000/predict";

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

clearBtn.onclick = () => {
  selectedFile = null;
  imagePreview.innerHTML = "No Image";
  fileName.textContent = "No file selected";
  resultDiv.innerHTML = "";
  imageInput.value = "";
};

predictBtn.onclick = async () => {
  if (!selectedFile) {
    resultDiv.innerHTML =
      "<p style='color: red;'>Please upload an image first.</p>";
    return;
  }

  resultDiv.innerHTML = "<p>Processing...</p>";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success && data.faces && data.faces.length > 0) {
      // Display results for all detected faces
      let resultHTML = `<h3>Detected ${data.num_faces_detected} face(s):</h3>`;

      data.faces.forEach((face, index) => {
        resultHTML += `
                    <div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                        <h4>Face ${face.face_index + 1}</h4>
                        <p><strong>Celebrity:</strong> ${
                          face.prediction.celebrity_name
                        }</p>
                        <p><strong>Confidence:</strong> ${(
                          face.prediction.confidence * 100
                        ).toFixed(2)}%</p>
                        <p><strong>Detection Confidence:</strong> ${(
                          face.detection_confidence * 100
                        ).toFixed(2)}%</p>
                        
                        <details>
                            <summary>Top 3 Predictions</summary>
                            <ol>
                                ${face.top_3_predictions
                                  .map(
                                    (pred) =>
                                      `<li>${pred.celebrity_name} (${(
                                        pred.confidence * 100
                                      ).toFixed(2)}%)</li>`
                                  )
                                  .join("")}
                            </ol>
                        </details>
                    </div>
                `;
      });

      resultDiv.innerHTML = resultHTML;
    } else if (!data.success) {
      resultDiv.innerHTML = `<p style='color: orange;'>${
        data.message || "No face detected in the image."
      }</p>`;
    } else {
      resultDiv.innerHTML =
        "<p style='color: red;'>Unexpected response from server.</p>";
    }
  } catch (error) {
    console.error("Error:", error);
    resultDiv.innerHTML = `<p style='color: red;'>Error connecting to backend: ${error.message}</p>`;
  }
};