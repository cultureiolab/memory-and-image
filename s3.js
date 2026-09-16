import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";

import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import dotenv from "dotenv";

dotenv.config();

// NEW: Added for 7-day presigned URL caching
const VIEW_URL_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
const REFRESH_BUFFER_MS = 24 * 60 * 60 * 1000;   // re-sign a day before it truly expires
const viewUrlCache = new Map(); // {url, expiresAt }


const s3 = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  },
});

// function for UPLOADING
export async function generatePresignedUrl(filename, contentType) {
  const command = new PutObjectCommand({
    Bucket: process.env.AWS_BUCKET_NAME,
    Key: filename,
    ContentType: contentType,
    CacheControl: "public, max-age=604800, immutable",

  });
  return await getSignedUrl(s3, command, { expiresIn: 300 });
}

// // function for VIEWING/DOWNLAODING
// export async function generateGetPresignedUrl(filename) {
//   const command = new GetObjectCommand({
//     Bucket: process.env.AWS_BUCKET_NAME,
//     Key: filename,
//   });

//   // This link will work for 1 hour (3600 seconds)
//   return await getSignedUrl(s3, command, { expiresIn: 3600 });
// }

export async function generateGetPresignedUrl(filename) {
  const cached = viewUrlCache.get(filename);
  const now = Date.now();
  if (cached && cached.expiresAt - now > REFRESH_BUFFER_MS) {
    return cached.url; // plenty of runway left -- reuse it
  }

  const command = new GetObjectCommand({
    Bucket: process.env.AWS_BUCKET_NAME,
    Key: filename,
  });
  const url = await getSignedUrl(s3, command, { expiresIn: 604800 }); // AWS's max for SigV4
  viewUrlCache.set(filename, { url, expiresAt: now + VIEW_URL_TTL_MS });
  return url;
}

// Uploads a Buffer (e.g. a server-generated thumbnail) directly to S3 
// no presigning needed since this runs server-side, not from the browser.
export async function uploadBuffer(filename, buffer, contentType) {
  const command = new PutObjectCommand({
    Bucket: process.env.AWS_BUCKET_NAME,
    Key: filename,
    Body: buffer,
    ContentType: contentType,
    CacheControl: "public, max-age=604800, immutable",
  });
  await s3.send(command);
}