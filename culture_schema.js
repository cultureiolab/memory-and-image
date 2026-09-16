import mongoose from 'mongoose';

const culture_schema = new mongoose.Schema({
    referralID: String,
    caption: {
        type: String,
        required: false,
        validate: {
            validator: function(v) {
                // Add forbidden words here
                const forbidden = []; 
                return !forbidden.some(word => v.toLowerCase().includes(word));
            },
            message: "Caption contains inappropriate language!"
        }
    },
    s3Url: String,
    thumbFilename: { type: String, default: null },
    location: {
        latitude: Number,
        longitude: Number,
    },
    contact: {
        type: String,
        required: [true, "Contact information is required to submit!"] // Now required
    },    
    approved: { type: Boolean, default: false }, // New submissions start as "Pending"
    timestamp: { type: Date, default: Date.now },
    imageId: { 
        type: String, 
        unique: true, 
    },
    // Restored taxonomy shape: { objects: [...], scene: { path: [...] }, primaryCategory }.
    // Filled in locally (inference.js) after upload.
    classification: {
        type: mongoose.Schema.Types.Mixed,
        default: null // null = not yet classified (used to find old entries needing backfill)
    },

    // CLIP image embedding (512-dim, L2-normalized), used ONLY for signature
    // matching via Atlas $vectorSearch (/api/webcam-match). Requires an Atlas
    // Vector Search index named "clipEmbedding_vector_index" on this field
    // (cosine similarity, 512 dimensions) -- see server.js comment.
    clipEmbedding: {
        type: [Number],
        default: null
    }

});

export const CultureModel = mongoose.model('Culture', culture_schema, 'Images');