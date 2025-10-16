const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const faceapi = require("face-api.js");
const canvas = require("canvas");

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Folders for storing images and descriptors
const FACE_DIR = path.resolve("./face");
const DESC_DIR = path.resolve("./descriptors");

const app = express();
const PORT = 3000;

app.use(cors({
  origin: "*",            // or ["http://10.249.38.61:4173"] for stricter setup
  methods: ["GET", "POST"],
  allowedHeaders: ["Content-Type"],
}));
app.use(express.json({ limit: "20mb" }));
app.use("/face", express.static(FACE_DIR));

// Make sure directories exist
if (!fs.existsSync(FACE_DIR)) fs.mkdirSync(FACE_DIR, { recursive: true });
if (!fs.existsSync(DESC_DIR)) fs.mkdirSync(DESC_DIR, { recursive: true });

let registrations = [];
// ==============================
// Load face-api.js models
// ==============================
const MODEL_PATH = path.resolve("./models");
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_PATH),
  faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH),
  faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH),
]).then(() => {
  console.log("✅ Models loaded");
  app.listen(PORT, "0.0.0.0", () =>
  console.log(`✅ Server running on http://10.249.38.91:${PORT}`)
);
}).catch(err => console.error("❌ Model load error:", err));

// ==============================
// Helper functions
// ==============================
function bufferFromBase64(base64) {
  return Buffer.from(base64.replace(/^data:image\/\w+;base64,/, ""), "base64");
}

function imageFromBase64(base64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = bufferFromBase64(base64);
  });
}

// TinyFaceDetector options
const tinyOptions = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.5
});

// ==============================
// Check if face already exists
// ==============================
async function isFaceAlreadyRegistered(descriptor) {
  const files = fs.readdirSync(DESC_DIR).filter(f => f.endsWith(".json"));

  for (let file of files) {
    const data = JSON.parse(fs.readFileSync(path.join(DESC_DIR, file)));

    for (let storedDesc of data.descriptors) {
      if (!storedDesc || storedDesc.length !== descriptor.length) {
        console.warn(`Skipping invalid descriptor in ${file}`);
        continue; // skip malformed descriptors
      }

      const dist = faceapi.euclideanDistance(descriptor, storedDesc);
      if (dist < 0.6) return true;
    }
  }

  return false;
}

// ==============================
// 🔹 Orientation checker
// ==============================
app.post("/check-face", async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.json({ orientation: "none" });

    const img = await imageFromBase64(image);
    const detection = await faceapi
      .detectSingleFace(img, tinyOptions)
      .withFaceLandmarks();

    if (!detection) return res.json({ orientation: "none" });

    const { landmarks } = detection;
    const nose = landmarks.getNose()[3];
    const leftEye = landmarks.getLeftEye()[0];
    const rightEye = landmarks.getRightEye()[3];

    const eyeDiff = rightEye.x - leftEye.x;
    const noseOffset = nose.x - (leftEye.x + eyeDiff / 2);

    let orientation = "front";
    if (noseOffset > 15) orientation = "left";
    if (noseOffset < -15) orientation = "right";

    res.json({ orientation });
  } catch (err) {
    console.error(err);
    res.json({ orientation: "none" });
  }
});
// ==============================
// Registration endpoint
// ==============================
app.post("/register", async (req, res) => {
  try {
    const { studentId, firstName, middleInitial, surname, images } = req.body;

    if (!studentId || !firstName || !surname || !images) {
      return res.status(400).json({ message: "Missing registration data" });
    }

    let descriptors = [];

    for (let key of ["pic1", "pic2", "pic3"]) {
      if (!images[key]) continue;

      // Convert base64 to image
      const img = await imageFromBase64(images[key]);

      // Detect face and get descriptor
      const detection = await faceapi
        .detectSingleFace(img, tinyOptions)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection || !detection.descriptor) continue;

      const desc = detection.descriptor;

      if (desc.length !== 128) continue; // skip invalid descriptors

      // Check if face already registered
      if (await isFaceAlreadyRegistered(desc)) {
        return res.status(400).json({ message: `❌ Face already registered!` });
      }

      // Save descriptor as normal array
      descriptors.push(Array.from(desc));

      // Save the actual image
      const imgBuffer = bufferFromBase64(images[key]);
      const imgFile = path.join(FACE_DIR, `${studentId}_${key}.png`);
      fs.writeFileSync(imgFile, imgBuffer);
    }

    if (descriptors.length === 0) {
      return res.status(400).json({ message: "❌ No valid face detected" });
    }

    // Save descriptors in JSON file
    const descFile = path.join(DESC_DIR, `${studentId}.json`);
    const data = { studentId, firstName, middleInitial, surname, descriptors };
    fs.writeFileSync(descFile, JSON.stringify(data, null, 2));

    return res.json({ message: `✔ Registered ${firstName} ${surname}` });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ message: "❌ Registration error", error: err.message });
  }
});


// ==============================
// Login / Recognition endpoint
// ==============================
app.post("/login-recognize", async (req, res) => {
  try {
    const { image } = req.body;
    if (!image) return res.status(400).json({ message: "Missing image" });

    const img = await imageFromBase64(image);
    const detection = await faceapi
      .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection || !detection.descriptor) {
      return res.json({ message: "❌ No face detected" });
    }

    const queryDesc = detection.descriptor;
    const files = fs.readdirSync(DESC_DIR).filter(f => f.endsWith(".json"));
    let bestMatch = null;
    let bestDist = 1.0;

    for (let file of files) {
      const data = JSON.parse(fs.readFileSync(path.join(DESC_DIR, file)));
      for (let storedDesc of data.descriptors) {
        if (!storedDesc || storedDesc.length !== queryDesc.length) continue;
        const dist = faceapi.euclideanDistance(queryDesc, storedDesc);
        if (dist < bestDist) {
          bestDist = dist;
          bestMatch = data;
        }
      }
    }

    if (bestMatch && bestDist < 0.6) {
      // Find all face images for this student
      const faceFiles = fs.readdirSync(FACE_DIR)
        .filter(f => f.includes(bestMatch.studentId));

      const imagePaths = faceFiles.map(f => `/face/${f}`);

      return res.json({
        message: `✅ Welcome back, ${bestMatch.firstName} ${bestMatch.surname}`,
        imagePaths, // send array of image URLs
        studentId: bestMatch.studentId // add this
      });
    }

    return res.json({ message: "❌ Stranger detected" });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ message: "❌ Recognition error", error: err.message });
  }
});