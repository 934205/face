const express = require("express");
const cors = require("cors");
const { createClient } = require("@supabase/supabase-js");
const faceapi = require("face-api.js");
const canvas = require("canvas");
const fetch = require("node-fetch");
require("dotenv").config()


const app = express();
app.use(cors());
app.use(express.json({ limit: "10mb" })); // allow large base64 images

const SUPABASE_URL = process.env.SUPABASE_URL
const SUPABASE_KEY = process.env.SUPABASE_KEY
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// Node canvas for face-api.js
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, fetch });

// Load models once
const MODEL_URL = "https://justadudewhohacks.github.io/face-api.js/models";
let modelsLoaded = false;

async function loadModels() {
  console.log("Loading face-api.js models from URL...");
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  modelsLoaded = true;
  console.log("Models loaded!");
}
loadModels();

// ✅ Endpoint: verify face
app.post("/verify", async (req, res) => {
  const { reg_no, selfieBase64 } = req.body;

  
  

  if (!reg_no || !selfieBase64) {
    return res.status(400).send("Missing reg_no or selfieBase64");
  }

  try {
    // Load selfie image from base64
    const base64Data = selfieBase64.replace(/^data:image\/\w+;base64,/, "");
    const buffer = Buffer.from(base64Data, "base64");
    const selfieImg = await canvas.loadImage(buffer);    

    // Load user image from Supabase
    const { data, error } = await supabase
      .from("student")
      .select("face_url")
      .eq("reg_no", reg_no)
      .single();

    if (error || !data) return res.status(404).send("User not found");
    const dbImg = await canvas.loadImage(data.face_url);

    // Detect faces
    const selfieDetection = await faceapi
      .detectSingleFace(selfieImg, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    const dbDetection = await faceapi
      .detectSingleFace(dbImg, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!selfieDetection || !dbDetection)
      return res.send({ match: false, distance: null });

    // Compare descriptors
    const distance = faceapi.euclideanDistance(
      selfieDetection.descriptor,
      dbDetection.descriptor
    );

    res.send({ match: distance < 0.5, distance });
  } catch (err) {
    console.error("Server error:", err);
    res.status(500).send("Server error");
  }
});

app.listen(5000)