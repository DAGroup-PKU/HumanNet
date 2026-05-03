import { publicPath } from "../lib/public-path";
import type { PerspectiveId } from "./perspectives";

export interface GalleryClip {
  id: string;
  src: string;
  perspective: PerspectiveId;
  task: string;
  caption: string;
  /** native aspect of the asset (w / h), measured with ffprobe */
  aspectRatio: number;
}

const TPV_CLIPS = [
  ["-WtHP8ippvA_35.mp4", 2134, 1280],
  ["0041f063a89d07febf14bca51749ee02.mp4", 2136, 1154],
  ["03d6283185194b2227d356077a3dbb1e.mp4", 2134, 1258],
  ["32saBdSl6zE_115.mp4", 2124, 528],
  ["3753dfcc015406616765644452c0edc3.mp4", 2134, 1280],
  ["3JQ9RgudzHA_9.mp4", 2134, 528],
  ["5dc9bffc84eab9923b4720889c339e92.mp4", 2134, 1196],
  ["8XFiCOFJies_67.mp4", 2134, 1300],
  ["C0-CyeDtBvw_111.mp4", 2134, 1238],
  ["FrcsD59EwiA_19.mp4", 2116, 528],
  ["PQ4HD4dzYWw_36.mp4", 2134, 1238],
  ["QzPCrOJJ1dg_19.mp4", 2134, 1216],
  ["R_r2Q054fsM_115.mp4", 2134, 1322],
  ["THOnoGJt2cA_724.mp4", 2404, 1300],
  ["ct5li5hUTJY_371.mp4", 2134, 528],
  ["ff06e3d175271ce533593256b6589112.mp4", 2136, 1174],
  ["kEOsGQVfiWs_35.mp4", 2154, 528],
  ["lwqLWQDonow_17.mp4", 2134, 1238],
  ["rqfTVyyu2As_39.mp4", 2134, 1216],
  ["wlGRqsphHfE_107.mp4", 2134, 1258],
] as const;

const FPV_CLIPS = [
  ["0001_0.4099_z176-sep-05-22-switch__864_1126_263_six_panel.mp4", "Switch handling", "First-person switch interaction, rendered as a six-panel research clip."],
  ["0011_0.3060_09f8cba3-0337-4043-b341-c241d5248cab__73800_74025_226_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0013_0.3327_R030-12July-Nespresso__3621_3794_174_six_panel.mp4", "Coffee machine", "Coffee-machine interaction, rendered as a six-panel egocentric clip."],
  ["0013_0.4577_factory_101_worker_013_0112__3407_3632_226_six_panel.mp4", "Factory work", "Factory workflow captured from the actor's viewpoint."],
  ["0013_0.8088_12147__56_154_99_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0017_0.3760_factory053_worker007_00090__510_635_126_six_panel.mp4", "Factory work", "Production-line task captured as a six-panel egocentric clip."],
  ["0020_0.3404_factory014_worker070_00028__1253_1581_329_six_panel.mp4", "Factory work", "Factory activity captured from a head-mounted perspective."],
  ["0022_0.3359_factory_103_worker_022_0122__3981_4173_193_six_panel.mp4", "Factory work", "Factory task sample with synchronized six-panel references."],
  ["0026_0.3115_z039-june-23-22-nespresso__5001_5261_261_six_panel.mp4", "Coffee machine", "Coffee-machine routine captured from the actor's viewpoint."],
  ["0026_0.3700_78920__59_172_114_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0026_0.5275_4ddc86f9-f3a5-4a0b-83f0-9c690b7cc1ad__38119_38247_129_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0031_0.3146_z025-june-21-22-printer_small__1138_1386_249_six_panel.mp4", "Office tool", "Office-printer interaction captured from an egocentric rig."],
  ["0032_0.5631_7abc5ff7-28a3-4904-a5e3-99c9b2e3c085__1251_1379_129_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0035_0.3738_85951__58_120_63_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0043_0.3280_factory039_worker006_00172__2431_2618_188_six_panel.mp4", "Factory work", "Factory workflow captured from the actor's viewpoint."],
  ["0044_0.3582_433fb84b-4f39-4d31-9975-3185d8a9bac1__38946_39103_158_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0045_0.4303_factory_134_worker_011_0059__817_990_174_six_panel.mp4", "Factory work", "Production-line task captured as a six-panel egocentric clip."],
  ["0047_0.3376_9a30454f-b06b-45b7-a8a0-90d3f4ea2eb1__25741_25920_180_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
  ["0048_0.3326_factory063_worker011_00065__2508_2672_165_six_panel.mp4", "Factory work", "Factory activity captured from a head-mounted perspective."],
  ["0050_0.3851_c2b3e7ea-fbeb-4491-b5bc-d0dd723496f7__62861_63006_146_six_panel.mp4", "Egocentric preview", "Curated egocentric HumanNet preview clip with six-panel rendering."],
] as const;

const tpvEntries = TPV_CLIPS.map(([file, width, height], index): GalleryClip => ({
  id: `tpv-preview-${String(index + 1).padStart(2, "0")}`,
  src: publicPath(`/media/curation/video/tpv/${file}`),
  perspective: "exo",
  task: "Third-person preview",
  caption: "Curated third-person HumanNet preview clip.",
  aspectRatio: width / height,
}));

const fpvEntries = FPV_CLIPS.map(([file, task, caption], index): GalleryClip => ({
  id: `fpv-preview-${String(index + 1).padStart(2, "0")}`,
  src: publicPath(`/media/curation/video/fpv/${file}`),
  perspective: "ego",
  task,
  caption,
  aspectRatio: 16 / 9,
}));

export const GALLERY: GalleryClip[] = [...tpvEntries, ...fpvEntries];
