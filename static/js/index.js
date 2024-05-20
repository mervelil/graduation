

  document.addEventListener("DOMContentLoaded", function () {
    const dots = document.querySelector(".dots");
    const uploadContainer = document.querySelector("#uploadContainer");
    const fileInput = document.querySelector("#fileInput");
    const uploadedImage = document.querySelector("#uploadedImage");

    dots.addEventListener("click", function () {
      uploadContainer.style.display = "flex";
    });

    fileInput.addEventListener("change", function (e) {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
          uploadedImage.src = event.target.result;
          uploadedImage.style.display = "block";
        };
        reader.readAsDataURL(file);
      }
    });
  });


  // const API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5";
  // const headers = {"Authorization": "Bearer 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"};
  
  // function query(payload) {
  //     return fetch(API_URL, {
  //         method: "POST",
  //         headers: headers,
  //         body: JSON.stringify(payload)
  //     }).then(response => {
  //         if (!response.ok) {
  //             throw new Error(`Request failed with status code ${response.status}`);
  //         }
  //         return response.blob();
  //     });
  // }
  
  // function updateImage(text) {
  //     query({ "text": text })
  //         .then(blob => {
  //             const currentImage = document.getElementById("currentImage");
  //             currentImage.src = URL.createObjectURL(blob);
  //         })
  //         .catch(error => {
  //             console.error("Failed to generate image:", error);
  //         });
  // }
  
  // // Define a list of inputs (texts) dynamically
  // const inputs = ["green dress"] * 7;  // Repeat the input "elbise istiyorum" 7 times
  
  // let currentIndex = 0;
  
  // function showNextImage() {
  //     updateImage(inputs[currentIndex]);
  //     currentIndex = (currentIndex + 1) % inputs.length;
  // }
  
  // // Update the image every 2 seconds (2000 milliseconds)
  // setInterval(showNextImage, 2000);
  



document.addEventListener("DOMContentLoaded", function() {
  var dots = document.getElementById("dots");
  var imageUpload = document.getElementById("imageUpload");
  var uploadedImageContainer = document.getElementById("uploadedImageContainer");
  var uploadedImage = document.getElementById("uploadedImage");

  dots.addEventListener("click", function() {
      imageUpload.click(); // Tıklama olayı tetiklendiğinde dosya seçme işlemi başlatılır
  });

  imageUpload.addEventListener("change", function() {
      var file = this.files[0]; // Seçilen dosya
      if (file) {
          var reader = new FileReader();
          reader.onload = function(event) {
              var imageData = event.target.result; // Dosya okunduğunda veri alınır
              uploadedImage.src = imageData; // Önizleme alanına resim eklenir
              uploadedImageContainer.style.display = "block"; // Önizleme alanı görünür hale getirilir
          };
          reader.readAsDataURL(file); // Dosya okuma işlemi başlatılır
      }
  });
});

