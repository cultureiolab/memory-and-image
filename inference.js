import ort from "onnxruntime-node";
import sharp from "sharp";
import { pipeline, RawImage } from "@xenova/transformers";
import path from "path";
import { readFile } from "fs/promises";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const YOLO_MODEL_PATH = path.join(__dirname, "models", "yolov8n-oiv7.onnx");
const YOLO_NAMES_PATH = path.join(__dirname, "models", "yolo_names.json");
const YOLO_INPUT_SIZE = 640;
const YOLO_CONF_THRESHOLD = 0.5; // matches the old conf=0.50 in classifier_service.py
const YOLO_IOU_THRESHOLD = 0.45; // standard NMS default


const TAXONOMY = {
  "an office": ["an indoor scene", "an office"],
  "a school or classroom": ["an indoor scene", "a school or classroom"],
  "a kitchen": ["an indoor scene", "a home interior", "a kitchen"],
  "a living room": ["an indoor scene", "a home interior", "a living room"],
  "a bedroom": ["an indoor scene", "a home interior", "a bedroom"],
  "a bathroom": ["an indoor scene", "a home interior", "a bathroom"],
  "a restaurant or cafe": ["an indoor scene", "a restaurant or cafe"],
  "a store or shopping mall": ["an indoor scene", "a store or shopping mall"],
  "a gym or fitness center": ["an indoor scene", "a gym or fitness center"],
  "a museum or art gallery": ["an indoor scene", "a museum or art gallery"],
  "an airport or train station": ["an indoor scene", "an airport or train station"],
  "a hospital or clinic": ["an indoor scene", "a hospital or clinic"],
  "a library": ["an indoor scene", "a library"],
  "a theater or concert hall": ["an indoor scene", "a theater or concert hall"],
  "a hotel room": ["an indoor scene", "a hotel room"],
  "a lake or body of water": ["an outdoor scene", "a lake or body of water"],
  "a beach or ocean": ["an outdoor scene", "a beach or ocean"],
  "mountains": ["an outdoor scene", "mountains"],
  "a desert": ["an outdoor scene", "a desert"],
  "a forest or hiking trail": ["an outdoor scene", "a forest or hiking trail"],
  "a busy city street": ["an outdoor scene", "a busy city street"],
  "a grass plain or field": ["an outdoor scene", "a grass plain or field"],
  "a park or garden": ["an outdoor scene", "a park or garden"],
  "farmland or countryside": ["an outdoor scene", "farmland or countryside"],
  "a stadium or sports field": ["an outdoor scene", "a stadium or sports field"],
  "a street": ["an outdoor scene", "a street"],
  "a bridge": ["an outdoor scene", "a bridge"],
  "sky view": ["an outdoor scene", "sky view"],
  "a backyard or patio": ["an outdoor scene", "a backyard or patio"],
  "a snowy or winter scene": ["an outdoor scene", "a snowy or winter scene"],
  "a computer, television, or phone screen": ["an indoor scene", "a computer, television, or phone screen"],
};
const LEAF_LABELS = Object.keys(TAXONOMY);
const CANDIDATE_LABELS = LEAF_LABELS.map(lbl => `a photo of ${lbl}`);

const MIN_SCENE_CONFIDENCE = 0.35; // below this, drop the subcategory, keep only indoor/outdoor
const NATURE_SUBCATEGORIES = new Set([
  "a lake or body of water", "a beach or ocean", "mountains", "a desert",
  "a forest or hiking trail", "a grass plain or field", "a park or garden",
  "farmland or countryside", "a snowy or winter scene",
]);

const GROUP_PHOTO_MIN_PEOPLE = 3;
const PERSON_LABELS = new Set(["Person", "Human face", "Human head"]); // matches PERSON_LABEL set in the original Python
const PET_LABELS = new Set(["Dog", "Cat", "Bird", "Rabbit", "Hamster", "Guinea pig"]);
const ARTWORK_LABELS = new Set(["Painting", "Picture frame", "Poster", "Sculpture"]);

let yoloSession = null;
let yoloNames = null; // { "0": "Person", "1": "Dog", ... }
let clipEmbeddingPipeline = null; // image-only embedding, used for webcam-match ($vectorSearch)
let clipClassificationPipeline = null; // zero-shot image-vs-text, used for the 31-label taxonomy

// Longest edge for the browser-facing display copy used by graph3d1's
// texture sprites. 1000px is comfortably sharp for the sprite sizes the
// graph actually renders at  while keeping each texture in the ~1-2MB range 
const DISPLAY_MAX_DIM = 1000;
const DISPLAY_JPEG_QUALITY = 82;

// Produces a resized, re-compressed copy of an uploaded photo for display
// in the 3D graph.
export async function makeDisplayThumbnail(imageBuffer) {
  return sharp(imageBuffer)
    .rotate() // apply EXIF orientation before resizing so sprites aren't sideways
    .resize(DISPLAY_MAX_DIM, DISPLAY_MAX_DIM, {
      fit: "inside",
      withoutEnlargement: true,
    })
    .jpeg({ quality: DISPLAY_JPEG_QUALITY })
    .toBuffer();
}

export async function loadModels() {
  const [session, namesRaw] = await Promise.all([
    ort.InferenceSession.create(YOLO_MODEL_PATH),
    readFile(YOLO_NAMES_PATH, "utf-8"),
  ]);
  yoloSession = session;
  yoloNames = JSON.parse(namesRaw);


  // image-feature-extraction = vision tower only (for matching),
  // zero-shot-image-classification = vision+text towers, scored against our candidate labels (for the taxonomy).
  clipEmbeddingPipeline = await pipeline("image-feature-extraction", "Xenova/clip-vit-base-patch32");
  clipClassificationPipeline = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32");

  console.log(`Loaded YOLO (${Object.keys(yoloNames).length} classes) and CLIP models`);
}

// Resizes+pads onto a 640x640 canvas without distorting aspect ratio
// (the standard YOLO "letterbox" preprocessing step), returns the raw
// RGB float tensor plus the scale/offset needed to map boxes back later.
async function letterbox(imageBuffer) {
  const img = sharp(imageBuffer).removeAlpha();
  const meta = await img.metadata();
  const scale = Math.min(YOLO_INPUT_SIZE / meta.width, YOLO_INPUT_SIZE / meta.height);
  const newW = Math.round(meta.width * scale);
  const newH = Math.round(meta.height * scale);
  const padX = Math.floor((YOLO_INPUT_SIZE - newW) / 2);
  const padY = Math.floor((YOLO_INPUT_SIZE - newH) / 2);

  const { data } = await img
    .resize(newW, newH)
    .extend({
      top: padY, bottom: YOLO_INPUT_SIZE - newH - padY,
      left: padX, right: YOLO_INPUT_SIZE - newW - padX,
      background: { r: 114, g: 114, b: 114 }, // YOLO's standard grey pad
    })
    .raw()
    .toBuffer({ resolveWithObject: true })
    .then(r => r);

  // HWC uint8 -> CHW float32, normalized 0..1
  const floatData = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
  const planeSize = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  for (let i = 0; i < planeSize; i++) {
    floatData[i] = data[i * 3] / 255;
    floatData[planeSize + i] = data[i * 3 + 1] / 255;
    floatData[2 * planeSize + i] = data[i * 3 + 2] / 255;
  }

  return { tensorData: floatData, scale, padX, padY, origW: meta.width, origH: meta.height };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function nms(boxes) {
  boxes.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const box of boxes) {
    if (kept.every(k => k.classId !== box.classId || iou(k, box) < YOLO_IOU_THRESHOLD)) {
      kept.push(box);
    }
  }
  return kept;
}

// Runs YOLO, returns the human-readable detections list (same shape as the
// old classifier_service.py "objects" field). No similarity vector anymore
// -- that was only needed for the now-removed clustering step.
export async function runYolo(imageBuffer) {
  const { tensorData, scale, padX, padY } = await letterbox(imageBuffer);
  const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);

  const inputName = yoloSession.inputNames[0];
  const outputName = yoloSession.outputNames[0];
  const results = await yoloSession.run({ [inputName]: inputTensor });
  const output = results[outputName]; // shape [1, 4+numClasses, 8400] for yolov8

  const numClasses = Object.keys(yoloNames).length;
  const numBoxes = output.dims[2];
  const data = output.data;

  const candidates = [];
  for (let i = 0; i < numBoxes; i++) {
    let bestClass = -1;
    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > bestScore) { bestScore = score; bestClass = c; }
    }
    if (bestScore < YOLO_CONF_THRESHOLD) continue;

    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    // undo letterbox padding/scaling to get real-image-space coords
    const x1 = (cx - w / 2 - padX) / scale;
    const y1 = (cy - h / 2 - padY) / scale;
    const x2 = (cx + w / 2 - padX) / scale;
    const y2 = (cy + h / 2 - padY) / scale;

    candidates.push({ classId: bestClass, confidence: bestScore, x1, y1, x2, y2 });
  }

  const kept = nms(candidates);

  const objects = kept.map(b => ({
    label: yoloNames[String(b.classId)],
    confidence: Math.round(b.confidence * 10000) / 10000,
  }));

  return { objects };
}

async function toRawImage(imageBuffer) {
  const blob = new Blob([imageBuffer]);
  return RawImage.fromBlob(blob);
}

// CLIP image embedding only -- used for webcam-match via Atlas $vectorSearch.
// Not used for grouping/clustering anymore.
export async function runClip(imageBuffer) {
  const rawImage = await toRawImage(imageBuffer);
  const output = await clipEmbeddingPipeline(rawImage, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// Restored taxonomy classification: scores the image against the 31 leaf
// labels (via CLIP's zero-shot image/text matching) and returns the same
// scene.path shape the original classifier_service.py produced.
export async function runSceneClassification(imageBuffer) {
  const rawImage = await toRawImage(imageBuffer);
  const results = await clipClassificationPipeline(rawImage, CANDIDATE_LABELS);
  // results: [{ label: "a photo of a kitchen", score: 0.62 }, ...] sorted desc
  const top = results[0];
  const bestLabel = top.label.replace(/^a photo of /, "");
  const bestScore = Math.round(top.score * 10000) / 10000;

  if (bestScore >= MIN_SCENE_CONFIDENCE) {
    return TAXONOMY[bestLabel].map(lvl => ({ label: lvl, confidence: bestScore }));
  }
  // Not confident enough to trust the specific subcategory -- keep only
  // the broad indoor/outdoor call from the same top match.
  return [{ label: TAXONOMY[bestLabel][0], confidence: bestScore }];
}

export function primaryCategoryFrom(objects, scenePath) {
  const detectedSet = new Set(objects.map(o => o.label));
  const personCount = objects.filter(o => PERSON_LABELS.has(o.label)).length;

  if (personCount >= GROUP_PHOTO_MIN_PEOPLE) return "group_photo";
  for (const label of detectedSet) if (PET_LABELS.has(label)) return "pet_photo";
  for (const label of detectedSet) if (ARTWORK_LABELS.has(label)) return "artwork_photo";
  if (!scenePath || !scenePath.length) return "unclassified";

  const topLevel = scenePath[0].label;
  const mostSpecific = scenePath[scenePath.length - 1].label;
  if (topLevel === "an outdoor scene") {
    return NATURE_SUBCATEGORIES.has(mostSpecific) ? "nature_photo" : "outdoor";
  }
  return topLevel === "an indoor scene" ? "indoor" : "unclassified";
}