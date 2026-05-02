// One-shot generator: emit TypeScript snippets for the 40-clip Preview
// gallery. Run with: `npx tsx scripts/gen-gallery.ts`.
//
// It is NOT used at runtime — the output is committed to
// server/src/default-config.ts and src/admin/admin-defaults.ts. Re-run
// when the curated clip set changes; copy the printed block into both
// destinations.

interface Clip {
  id: string;
  key: string;
  perspective: "exo" | "ego";
  task: string;
  caption: string;
  aspectRatio: number;
}

const TPV_KOALA_PREFIX =
  "public/dataset/third_person/Koala-36M-v1/rendered/Koala-36M-v1_sub_001/";
const TPV_OHV_PREFIX =
  "public/dataset/third_person/OpenHumanVid/rendered/OpenHumanVid_sub_001/";
const FPV_PREFIX = "data/dataset/egocentric/show_cases/selected/";

// Curated 10 Koala-36M clips (sampled to cover several `part_*` shards).
const KOALA_CLIPS: Array<{ part: string; file: string; task: string; caption: string }> = [
  { part: "Koala-36M-v1_sub_001_part_0001", file: "GQaByBr_QO0_64.mp4", task: "Mobile rig", caption: "Hand-held gimbal capture under mixed indoor lighting." },
  { part: "Koala-36M-v1_sub_001_part_0001", file: "HnjaKDRnmFM_64.mp4", task: "Studio rig", caption: "Multi-shot studio sequence from the Koala-36M montage subset." },
  { part: "Koala-36M-v1_sub_001_part_0001", file: "QSuxYRr3n7o_28.mp4", task: "Manipulation", caption: "Bimanual pick-and-place under a fixed studio rig." },
  { part: "Koala-36M-v1_sub_001_part_0001", file: "QSuxYRr3n7o_68.mp4", task: "Navigation", caption: "Cluttered indoor traversal with dynamic obstacles." },
  { part: "Koala-36M-v1_sub_001_part_0001", file: "QSuxYRr3n7o_80.mp4", task: "Assembly", caption: "Multi-step part-fitting on a workbench." },
  { part: "Koala-36M-v1_sub_001_part_0001", file: "QSuxYRr3n7o_101.mp4", task: "Tool use", caption: "Hand-tool sequencing under occlusion." },
  { part: "Koala-36M-v1_sub_001_part_0006", file: "KnMVTT5EcXM_14.mp4", task: "Workshop", caption: "Workshop activity recorded in a bench-top capture rig." },
  { part: "Koala-36M-v1_sub_001_part_0007", file: "WnJbxVE9jJE_43.mp4", task: "Studio rig", caption: "Multi-actor studio capture, mid-distance framing." },
  { part: "Koala-36M-v1_sub_001_part_0009", file: "9UkZ-mmDXAg_0.mp4", task: "Daily activity", caption: "Routine daily-life action under a fixed studio camera." },
  { part: "Koala-36M-v1_sub_001_part_0011", file: "K4aoW7LCDK0_167.mp4", task: "Daily activity", caption: "Single-actor daily activity from the Koala-36M corpus." },
];

// Curated 10 OpenHumanVid clips (one per part_002x shard, mid-aspect).
const OHV_CLIPS: Array<{ part: string; file: string; task: string; caption: string }> = [
  { part: "OpenHumanVid_sub_001_part_0021", file: "14486e0ba62338990040b634e052900b.mp4", task: "Crowd scene", caption: "Crowd-sourced human activity from the OpenHumanVid corpus." },
  { part: "OpenHumanVid_sub_001_part_0021", file: "14b65a2bc1a17f9cce0f6abb9853d343.mp4", task: "Daily activity", caption: "Wide-shot daily activity, mid-scene framing." },
  { part: "OpenHumanVid_sub_001_part_0022", file: "1523721c60d01bcf6b9461532008a00e.mp4", task: "Outdoor scene", caption: "Outdoor sequence from the OpenHumanVid set." },
  { part: "OpenHumanVid_sub_001_part_0022", file: "1570bbe5e5eb9c0710abb9525bcef8ca.mp4", task: "Indoor scene", caption: "Indoor crowd-sourced clip, mid-distance framing." },
  { part: "OpenHumanVid_sub_001_part_0023", file: "16908c584f98bc9a02526131ba9601fc.mp4", task: "Long take", caption: "Extended single-camera take, multi-actor." },
  { part: "OpenHumanVid_sub_001_part_0023", file: "16e13b754e967eea1dc187875b789e1a.mp4", task: "Daily activity", caption: "Routine human activity, OpenHumanVid sample." },
  { part: "OpenHumanVid_sub_001_part_0024", file: "175b2709b6a92b1e0556228ebd921b09.mp4", task: "Indoor scene", caption: "Indoor activity clip, OpenHumanVid sample." },
  { part: "OpenHumanVid_sub_001_part_0025", file: "184fe60ca3642a57c95a1df666533550.mp4", task: "Wide shot", caption: "Wide-angle human-activity capture." },
  { part: "OpenHumanVid_sub_001_part_0025", file: "18a3f9ba80e256ab15740ab1abff85a3.mp4", task: "Crowd scene", caption: "Multi-person scene from the OpenHumanVid corpus." },
  { part: "OpenHumanVid_sub_001_part_0026", file: "1992c6bdec3f8731069439bbbda15a25.mp4", task: "Indoor scene", caption: "Indoor session, OpenHumanVid sample." },
];

// Curated 20 egocentric six-panel clips, sampled across factory /
// nespresso / assembly / archival contexts.
const FPV_CLIPS: Array<{ file: string; task: string; caption: string }> = [
  { file: "0001_0.4099_z176-sep-05-22-switch__864_1126_263_six_panel.mp4", task: "Switch handling", caption: "First-person panel-switch operation, daylight rig." },
  { file: "0001_0.4390_fbe0a401-9bf3-4725-bddb-b2c44472cda0__4312_4473_162_six_panel.mp4", task: "Archival run", caption: "Helmet-rig archival sequence with multi-camera sync." },
  { file: "0001_0.5579_93414__33_191_159_six_panel.mp4", task: "Bench take", caption: "Numbered ego take from the open-bench capture set." },
  { file: "0003_0.4354_z088-july-07-22-dslr__498_873_376_six_panel.mp4", task: "DSLR rig", caption: "Outdoor head-mounted DSLR capture." },
  { file: "0004_0.3625_z140-aug-16-22-gopro__855_1012_158_six_panel.mp4", task: "GoPro rig", caption: "GoPro-mounted egocentric workshop capture." },
  { file: "0006_0.3873_63009__44_122_79_six_panel.mp4", task: "Bench take", caption: "Egocentric bench take with six-panel rendering." },
  { file: "0006_0.4284_factory_005_worker_011_0069__0_176_177_six_panel.mp4", task: "Factory work", caption: "Six-panel ego sweep on a production line." },
  { file: "0008_0.4179_25123__47_129_83_six_panel.mp4", task: "Workshop sweep", caption: "Workshop ego sweep with tool-set close-ups." },
  { file: "0010_0.5277_z028-june-22-22-marius_assemble__5373_5505_133_six_panel.mp4", task: "Assembly", caption: "Studio-rigged assembly task, ego perspective." },
  { file: "0011_0.3060_09f8cba3-0337-4043-b341-c241d5248cab__73800_74025_226_six_panel.mp4", task: "Archival run", caption: "Long ego archival sequence, multi-camera sync." },
  { file: "0012_0.3559_z035-june-23-22-rashult_assemble__9188_9362_175_six_panel.mp4", task: "Assembly", caption: "Studio-rig assembly task with hand-pose tracking." },
  { file: "0013_0.3327_R030-12July-Nespresso__3621_3794_174_six_panel.mp4", task: "Coffee machine", caption: "Coffee-machine interaction, head-mounted ego rig." },
  { file: "0013_0.3507_factory011_worker035_00053__2315_2548_234_six_panel.mp4", task: "Factory work", caption: "Production-line worker capture, ego six-panel." },
  { file: "0013_0.4577_factory_101_worker_013_0112__3407_3632_226_six_panel.mp4", task: "Factory work", caption: "Factory bench task, ego perspective." },
  { file: "0017_0.3073_z132-aug-12-22-nespresso__2955_3120_166_six_panel.mp4", task: "Coffee machine", caption: "Espresso preparation captured on a head-mounted rig." },
  { file: "0017_0.3760_factory053_worker007_00090__510_635_126_six_panel.mp4", task: "Factory work", caption: "Worker-side production capture, ego six-panel." },
  { file: "0026_0.3115_z039-june-23-22-nespresso__5001_5261_261_six_panel.mp4", task: "Coffee machine", caption: "Coffee-machine routine, head-mounted ego." },
  { file: "0028_0.3739_R086-27July-Nespresso__1193_1319_127_six_panel.mp4", task: "Coffee machine", caption: "Repeat coffee-prep capture, ego six-panel." },
  { file: "0031_0.3146_z025-june-21-22-printer_small__1138_1386_249_six_panel.mp4", task: "Office tool", caption: "Office-printer interaction from the ego studio rig." },
  { file: "0050_0.3851_c2b3e7ea-fbeb-4491-b5bc-d0dd723496f7__62861_63006_146_six_panel.mp4", task: "Archival run", caption: "Long archival ego sequence with multi-camera sync." },
];

const clips: Clip[] = [];

KOALA_CLIPS.forEach((c, i) => {
  clips.push({
    id: `tpv-koala-${String(i + 1).padStart(2, "0")}`,
    key: `${TPV_KOALA_PREFIX}${c.part}/${c.file}`,
    perspective: "exo",
    task: c.task,
    caption: c.caption,
    aspectRatio: 2134 / 1280,
  });
});

OHV_CLIPS.forEach((c, i) => {
  clips.push({
    id: `tpv-ohv-${String(i + 1).padStart(2, "0")}`,
    key: `${TPV_OHV_PREFIX}${c.part}/${c.file}`,
    perspective: "exo",
    task: c.task,
    caption: c.caption,
    aspectRatio: 16 / 9,
  });
});

FPV_CLIPS.forEach((c, i) => {
  clips.push({
    id: `fpv-ego-${String(i + 1).padStart(2, "0")}`,
    key: `${FPV_PREFIX}${c.file}`,
    perspective: "ego",
    task: c.task,
    caption: c.caption,
    aspectRatio: 16 / 9,
  });
});

const aspectLiteral = (n: number): string => {
  if (Math.abs(n - 16 / 9) < 1e-6) return "16 / 9";
  if (Math.abs(n - 2134 / 1280) < 1e-6) return "2134 / 1280";
  return n.toFixed(6);
};

const indent = "    ";
console.log("// ──────────────────── BEGIN GENERATED ────────────────────");
for (const c of clips) {
  console.log(`${indent}{`);
  console.log(`${indent}  id: ${JSON.stringify(c.id)},`);
  console.log(`${indent}  src: { kind: "oss" as const, key: ${JSON.stringify(c.key)} },`);
  console.log(`${indent}  perspective: ${JSON.stringify(c.perspective)} as const,`);
  console.log(`${indent}  task: ${JSON.stringify(c.task)},`);
  console.log(`${indent}  caption: ${JSON.stringify(c.caption)},`);
  console.log(`${indent}  aspectRatio: ${aspectLiteral(c.aspectRatio)},`);
  console.log(`${indent}},`);
}
console.log("// ──────────────────── END GENERATED ────────────────────");
console.log(`// total: ${clips.length} clips`);
