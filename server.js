import express from "express";
import cors from "cors";
import crypto from "crypto";
import mongoose from "mongoose";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "node:url";
import basicAuth from "express-basic-auth";

import { GraphModel } from "./graph_schema.js";


import { generatePresignedUrl, generateGetPresignedUrl, uploadBuffer } from "./s3.js";
import { CultureModel } from "./culture_schema.js";
import { loadModels, runYolo, runClip, runSceneClassification, primaryCategoryFrom, makeDisplayThumbnail } from "./inference.js";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log("Connected to MongoDB Atlas"))
  .catch(err => console.error("MongoDB connection error:", err));

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "/frontend")));

const adminProtector = basicAuth({
    users: { [process.env.ADMIN_USER || 'admin']: process.env.ADMIN_PASS || 'password123' }, 
    challenge: true,
    realm: 'CultureIO Admin'
});

const port = process.env.PORT || 3000;

// Name of the Atlas Vector Search index you create in the Atlas UI/CLI on
// the "clipEmbedding" field (cosine similarity, 512 dimensions). Nothing
// in this file creates the index -- see culture_schema.js comment.
const CLIP_VECTOR_INDEX = process.env.CLIP_VECTOR_INDEX || "clipEmbedding_vector_index";
const WEBCAM_MATCH_MIN_SCORE = Number(process.env.WEBCAM_MATCH_MIN_SCORE || 0.90); // needs calibration against real captures

// Same filename-resolution fallback used in /api/admin/pending, so this
// stays consistent whether an entry has an s3Url or only an imageId.
function resolveFilename(doc) {
  if (doc.s3Url) return doc.s3Url.split("/").pop();
  if (doc.imageId) return `${doc.imageId}.jpeg`;
  return null;
}

async function fetchImageBuffer(filename) {
  const imageUrl = await generateGetPresignedUrl(filename);
  const res = await fetch(imageUrl);
  if (!res.ok) throw new Error(`Could not download image (status ${res.status})`);
  return Buffer.from(await res.arrayBuffer());
}

// Derives the S3 key for a photo's display-sized copy from its original
// filename, e.g. "abc123.jpeg" -> "thumbs/abc123.jpg". 
function thumbFilenameFor(originalFilename) {
  const base = originalFilename.replace(/\.[^./]+$/, "");
  return `thumbs/${base}.jpg`;
}

// Generates the resized display copy and uploads it to S3. 
// Failure here is noted but won't block classification
// worst case the graph keeps falling back to the full-res original for this one photo.
async function makeAndUploadThumbnail(doc, filename, imageBuffer) {
  try {
    const thumbBuffer = await makeDisplayThumbnail(imageBuffer);
    const thumbFilename = thumbFilenameFor(filename);
    await uploadBuffer(thumbFilename, thumbBuffer, "image/jpeg");
    await CultureModel.findByIdAndUpdate(doc._id, { thumbFilename });
  } catch (err) {
    console.error(`Thumbnail generation failed for ${doc._id}:`, err.message);
  }
}

// Runs YOLO + CLIP taxonomy classification + CLIP embedding locally (no
// more Python microservice calls) and saves classification + clipEmbedding
// in one write. clipEmbedding is kept ONLY for /api/webcam-match now.
async function classifyAndSave(doc) {
  const filename = resolveFilename(doc);
  if (!filename) {
    console.error(`Classification skipped for ${doc._id}: no s3Url/imageId on this entry`);
    return;
  }
  try {
    const imageBuffer = await fetchImageBuffer(filename);

    const [{ objects }, scenePath, clipEmbedding] = await Promise.all([
      runYolo(imageBuffer),
      runSceneClassification(imageBuffer),
      runClip(imageBuffer),
      makeAndUploadThumbnail(doc, filename, imageBuffer), // same downloaded buffer
    ]);
    const primaryCategory = primaryCategoryFrom(objects, scenePath);

    await CultureModel.findByIdAndUpdate(doc._id, {
      classification: { objects, scene: { path: scenePath }, primaryCategory },
      clipEmbedding,
    });
  } catch (err) {
    console.error(`Classification failed for ${doc._id}:`, err.message, err.cause || "");
  }
}

// gets the presigned URLs for all nodes in a graph, so the frontend can display them
async function mindGraphImg(graph) {
  const plainGraph = graph.toObject ? graph.toObject() : graph;
  const docs = await CultureModel.find({ imageId: { $in: plainGraph.nodes } }).lean();
 
  console.log(`mindGraphImg: requested ${plainGraph.nodes.length} node(s), found ${docs.length} matching doc(s) in MongoDB`);
  const foundIds = new Set(docs.map((d) => d.imageId));
  plainGraph.nodes.forEach((id) => {
    if (!foundIds.has(id)) console.log(`  NO MATCHING DOCUMENT for imageId: ${id}`);
  });
 
  const nodeImages = {};
  await Promise.all(
    docs.map(async (doc) => {
      const filename = resolveFilename(doc);
      console.log(`  ${doc.imageId} -> resolveFilename: ${filename}`);
      if (!filename) {
        console.log(`  SKIPPED ${doc.imageId}: resolveFilename returned null (no s3Url or imageId on this doc)`);
        return;
      }
      try {
        nodeImages[doc.imageId] = await generateGetPresignedUrl(filename);
      } catch (err) {
        console.error(`  S3 sign FAILED for ${doc.imageId} (filename: ${filename}):`, err.message);
      }
    })
  );
 
  return { ...plainGraph, nodeImages };
}
 
 

// STEP 1: Request an upload "Ticket"
app.post("/api/get-upload-url", async (req, res) => {
  try {
    // Receive the custom fileName from the frontend
    const { contentType, fileName } = req.body;
    if (!fileName) {
      return res.status(400).json({ error: "fileName is required" });
    }
    const uploadUrl = await generatePresignedUrl(fileName, contentType);
    // Construct the public URL using the bucket name and the new filename
    const publicUrl = `https://${process.env.AWS_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${fileName}`;

    res.json({ uploadUrl, publicUrl });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// STEP 2: Save and return the specific image data
app.post("/api/save-entry", async (req, res) => {
  try {
    // 1. You MUST extract these from req.body
    const { 
      referralID,  
      s3Url, 
      timestamp, 
      location, 
      contact, 
      caption,
      imageId
    } = req.body;

    // 2. Pass them into the model
    const newEntry = new CultureModel({
      referralID, // Ensure this matches your schema (referralID vs nfcTagId)
      s3Url,
      timestamp,
      location,
      contact,
      caption,
      imageId
    });

    await newEntry.save();
    // fire-and-forget: don't make the user wait on inference
    classifyAndSave(newEntry);
    res.json({ success: true });
  } catch (err) {
    console.error("Database Save Error:", err); // This prints the REAL error to your terminal
    res.status(500).json({ error: err.message });
  }
});

// admin moderation
app.get("/admin.html", adminProtector, (req, res) => {
    res.sendFile(path.join(__dirname, "/frontend/admin.html"));
});

app.get("/api/admin/pending", adminProtector, async (req, res) => {
  try {
    const data = await CultureModel.find({ approved: false }).sort({ createdAt: -1 });
    res.json(data);
  } catch (err) {
    console.error("Admin route crash:", err);
    res.status(500).json({ error: err.message });
  }
});

app.put("/api/admin/approve/:id", adminProtector, async (req, res) => {
  await CultureModel.findByIdAndUpdate(req.params.id, { approved: true });
  res.json({ success: true });
});

app.delete("/api/admin/delete/:id", adminProtector, async (req, res) => {
  await CultureModel.findByIdAndDelete(req.params.id);
  res.json({ success: true });
});

// Single backfill route -- one inference pass produces classification AND
// clipEmbedding together, so there's still no separate signature-backfill.
app.post("/api/admin/classify-backfill", adminProtector, async (req, res) => {
  const force = req.query.force === "true";
  const query = force
    ? {}
    : { $or: [{ classification: null }, { classification: { $exists: false } }] };

  const targets = await CultureModel.find(query);
  console.log(`Backfill starting: ${targets.length} photos to process`);
  for (let i = 0; i < targets.length; i++) {
    await classifyAndSave(targets[i]); // sequential: gentle on memory/CPU during a big run
    console.log(`Backfill progress: ${i + 1}/${targets.length}`);
  }
  console.log("Backfill complete");
  res.json({ success: true, processed: targets.length, force });
});

// Public: data for the graphing pages (graph3d.html, graphcytoscape.html, graphscratch.html)
app.get("/api/graph-data", async (req, res) => {
  try {
    // const docs = await CultureModel.find({ approved: true }).lean();
    const docs = await CultureModel.find({}).lean();
 
    const imageNodes = await Promise.all(docs.map(async (doc) => {
      // Prefer the resized display copy so the graph's sprite textures stay small
      // only fall back to the full-res original for photos that haven't finished their first classification pass yet.
      const displayFilename = doc.thumbFilename || resolveFilename(doc);
      const originalFilename = resolveFilename(doc);
      let viewUrl = null;
      let fullUrl = null;
      try {
        if (displayFilename) viewUrl = await generateGetPresignedUrl(displayFilename);
        // Only sign the original separately when it's actually a different
        // object than what's already being used as the display copy.
        fullUrl = (originalFilename && originalFilename !== displayFilename)
          ? await generateGetPresignedUrl(originalFilename)
          : viewUrl;
      } catch (err) {
        console.error(`S3 Sign failed for ${displayFilename || originalFilename}:`, err.message);
      }

      const scenePath = doc.classification?.scene?.path || [];
      const sceneBroad = doc.classification?.primaryCategory || scenePath[0]?.label || "unclassified";
      const sceneSpecific = scenePath[scenePath.length - 1]?.label || sceneBroad;
      const sceneConfidence = scenePath[0]?.confidence ?? 0;

      return {
        id: doc._id.toString(),
        imageId: doc.imageId,
        img: viewUrl || 'https://via.placeholder.com/150?text=No+Image+Reference',
        fullImg: fullUrl || viewUrl || 'https://via.placeholder.com/150?text=No+Image+Reference',
        caption: doc.caption || "",
        timestamp: doc.timestamp,
        isHub: false,
        sceneBroad,
        sceneSpecific,
        sceneConfidence,
      };
    }));

    const groupIds = [...new Set(imageNodes.map(n => n.sceneBroad))];
    const hubNodes = groupIds.map(gid => {
      const members = imageNodes.filter(n => n.sceneBroad === gid);
      // highest-confidence member represents the hub, matching the original behavior
      const representative = members.reduce((best, n) =>
        (!best || n.sceneConfidence > best.sceneConfidence) ? n : best, null);

      return {
        id: `hub:${gid}`,
        isHub: true,
        label: gid,
        img: representative?.img || null,
        representativeId: representative?.id || null,
      };
    });
 
    const edges = imageNodes.map(n => ({
      source: n.id,
      target: `hub:${n.sceneBroad}`,
      confidence: n.sceneConfidence,
    }));
 
    res.json({ nodes: [...hubNodes, ...imageNodes], edges });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Signature matching via Atlas Vector Search -- replaces the old ORB
// BFMatcher loop entirely. One aggregation call does the nearest-neighbor
// search server-side instead of pulling descriptor blobs into Node memory.
app.post("/api/webcam-match", async (req, res) => {
  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) {
      return res.status(400).json({ error: "imageBase64 is required" });
    }

    const raw = imageBase64.split(",").pop(); // tolerate "data:image/jpeg;base64,..." prefixes
    const imageBuffer = Buffer.from(raw, "base64");
    const queryEmbedding = await runClip(imageBuffer);

    const results = await CultureModel.aggregate([
      {
        $vectorSearch: {
          index: CLIP_VECTOR_INDEX,
          path: "clipEmbedding",
          queryVector: queryEmbedding,
          numCandidates: 100,
          limit: 1,
          filter: { },
        },
      },
      {
        $project: {
          _id: 1,
          score: { $meta: "vectorSearchScore" },
        },
      },
    ]);

    const best = results[0];
    if (!best || best.score < WEBCAM_MATCH_MIN_SCORE) {
      return res.json({ matchId: null, score: best?.score ?? 0, reason: "below confidence threshold" });
    }

    res.json({ matchId: best._id.toString(), score: best.score });
  } catch (err) {
    console.error("Webcam match failed:", err.message, err.cause || "");
    res.status(500).json({ error: err.message });
  }
});


// Save a new proximity-graph capture (called from mindar-test.html on space-press)
 
app.post("/api/graphs", async (req, res) => {
  try {
    const { nodes, edges } = req.body;
 
    if (!Array.isArray(nodes) || nodes.length === 0) {
      return res.status(400).json({ error: "nodes must be a non-empty array" });
    }
    if (!Array.isArray(edges)) {
      return res.status(400).json({ error: "edges must be an array" });
    }
 
    const graph = new GraphModel({ nodes, edges });
    await graph.save();
 
    const getImg = await mindGraphImg(graph);
    res.json({ success: true, graph: getImg });
  } catch (err) {
    console.error("Save graph failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});

 
app.get("/api/graphs", async (req, res) => {
  try {
    const graphs = await GraphModel.find({}).sort({ createdAt: 1 }).lean();
    const getImg = await Promise.all(graphs.map(mindGraphImg));
    res.json(getImg);
  } catch (err) {
    console.error("List graphs failed:", err.message);
    res.status(500).json({ error: err.message });
  }
});
 


loadModels()
  .then(() => {
    app.listen(port, "0.0.0.0", () => {
      console.log(`Server running on port ${port}`);
    });
  })
  .catch(err => {
    console.error("Failed to load inference models, not starting server:", err);
    process.exit(1);
  });

// the url on the tag will be the regular url plus /tagIdname
app.get("/:tagId", (req, res, next) => {
    // If the request is for a file (like style.css or script.js), skip this
    if (path.extname(req.params.tagId)) {
        return next();
    }
    res.sendFile(path.join(__dirname, "/frontend/index.html"));
});