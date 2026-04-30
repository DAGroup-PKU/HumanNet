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

// EXO sources (varied research montages, avg ~5:3): 2134×{1216..1322}
// EGO sources (uniform six-panel, 16:9): 1280×720
export const GALLERY: GalleryClip[] = [
  {
    id: "exo-manip",
    src: "/videos/exo/QSuxYRr3n7o_28.mp4",
    perspective: "exo",
    task: "Manipulation",
    caption: "Bimanual pick-and-place under a fixed studio rig.",
    aspectRatio: 2134 / 1280,
  },
  {
    id: "exo-nav",
    src: "/videos/exo/QSuxYRr3n7o_68.mp4",
    perspective: "exo",
    task: "Navigation",
    caption: "Cluttered indoor traversal with dynamic obstacles.",
    aspectRatio: 2134 / 1300,
  },
  {
    id: "exo-assembly",
    src: "/videos/exo/QSuxYRr3n7o_80.mp4",
    perspective: "exo",
    task: "Assembly",
    caption: "Multi-step part-fitting on a workbench.",
    aspectRatio: 2134 / 1322,
  },
  {
    id: "exo-tool",
    src: "/videos/exo/QSuxYRr3n7o_101.mp4",
    perspective: "exo",
    task: "Tool use",
    caption: "Hand-tool sequencing under occlusion.",
    aspectRatio: 2134 / 1280,
  },
  {
    id: "ego-factory",
    src: "/videos/ego/0006_0.4284_factory_005_worker_011_0069__0_176_177_six_panel.mp4",
    perspective: "ego",
    task: "Factory work",
    caption: "Six-panel ego sweep on a production line.",
    aspectRatio: 16 / 9,
  },
  {
    id: "ego-switch",
    src: "/videos/ego/0001_0.4099_z176-sep-05-22-switch__864_1126_263_six_panel.mp4",
    perspective: "ego",
    task: "Switch handling",
    caption: "First-person panel-switch operation, daylight rig.",
    aspectRatio: 16 / 9,
  },
  {
    id: "ego-dslr",
    src: "/videos/ego/0003_0.4354_z088-july-07-22-dslr__498_873_376_six_panel.mp4",
    perspective: "ego",
    task: "DSLR rig",
    caption: "Outdoor head-mounted DSLR capture.",
    aspectRatio: 16 / 9,
  },
  {
    id: "ego-gopro",
    src: "/videos/ego/0004_0.3625_z140-aug-16-22-gopro__855_1012_158_six_panel.mp4",
    perspective: "ego",
    task: "GoPro rig",
    caption: "GoPro-mounted egocentric workshop capture.",
    aspectRatio: 16 / 9,
  },
];
