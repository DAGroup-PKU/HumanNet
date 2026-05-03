// Compatibility helper for the old `gen-gallery` command.
//
// The Preview gallery is now manifest-driven. Edit
// server/src/preview-gallery.ts, then run:
//
//   npm --prefix server run check:gallery
//
// This script prints the current rendered gallery for quick inspection.

import { PREVIEW_GALLERY } from "../src/preview-gallery.js";

console.log(JSON.stringify(PREVIEW_GALLERY, null, 2));
