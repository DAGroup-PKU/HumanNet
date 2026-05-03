import { Card } from "@heroui/react";
import { Section } from "./Section";
// Charts ship as WebP — quality 88 (cwebp) on the original PNGs, q82
// on the JPEG dashboard. Total ~2.8 MB across all 7, vs ~8.7 MB for
// the source PNG/JPEG versions. WebP has full coverage in evergreen
// browsers (Safari ≥14, Chrome ≥32, Firefox ≥65), so no <picture>
// fallback is needed for this audience.
import sceneSunburstUrl from "../assets/FPV/scene_distribution_sunburst.webp";
import sceneBarUrl from "../assets/FPV/scene_distribution_bar.webp";
import egoObjectUrl from "../assets/FPV/ego_object.webp";
import egoTaskUrl from "../assets/FPV/ego_task.webp";
import exoMotionUrl from "../assets/FPV/exo_motion_distribution.webp";
import motionSunburstUrl from "../assets/TPV/motion_categories_sunburst.webp";
import motionDashboardUrl from "../assets/TPV/motion_quality_dashboard.webp";

interface ChartCardProps {
  src: string;
  alt: string;
  tag: string;
  title: string;
  caption: string;
  /** Optional aspect ratio hint so the placeholder reserves the right
   *  amount of vertical space before the PNG decodes — prevents layout
   *  shift on slow connections. */
  aspectRatio?: string;
}

function ChartCard({ src, alt, tag, title, caption, aspectRatio }: ChartCardProps) {
  return (
    <Card
      variant="default"
      className="overflow-hidden bg-nebula-surface ring-1 ring-inset ring-nebula-line transition-colors hover:ring-nebula-line-strong"
    >
      <div className="flex items-center justify-between gap-3 px-1">
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-primary">
          {tag}
        </span>
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-nebula-on-dim">
          live · public
        </span>
      </div>
      <div
        className="mt-3 overflow-hidden rounded-md bg-white"
        style={aspectRatio ? { aspectRatio } : undefined}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          className="h-full w-full object-contain"
          draggable={false}
        />
      </div>
      <Card.Header className="mt-4">
        <Card.Title className="font-display text-lg text-nebula-on">
          {title}
        </Card.Title>
        <Card.Description className="text-sm leading-relaxed text-nebula-on-muted">
          {caption}
        </Card.Description>
      </Card.Header>
    </Card>
  );
}

export function DatasetProfile() {
  return (
    <Section
      id="profile"
      eyebrow="Dataset · profile"
      title={
        <>
          A first look at{" "}
          <span className="text-nebula-primary">what's inside HumanNet.</span>
        </>
      }
      description={
        <>
          Distributions, taxonomies, and quality signals from the Preview
          release. Every chart is rendered from the same metadata files
          shipped with the dataset, so researchers can reproduce them with the
          accompanying metadata scripts.
        </>
      }
    >
      <div className="grid gap-4">
        {/* Row 1 — Hero shot: scene-coverage sunburst.
            This chart compresses the most "is the data diverse?" signal
            into a single glance, so it earns full width on every
            breakpoint. */}
        <ChartCard
          src={sceneSunburstUrl}
          alt="Sunburst chart of scene coverage across Home, Industrial, Work, Outdoor, Retail, Transportation, Dining, Medical, Sports, and Other categories"
          tag="01 · scene coverage"
          title="30+ scene categories, three-level taxonomy."
          caption="Home dominates at 53.9% (kitchen / living-room / bedroom), with the long tail covering factories, offices, transportation, and outdoor work — the kind of breadth embodied agents need to generalize beyond controlled lab settings."
          aspectRatio="1024 / 760"
        />

        {/* Row 2 — Object & task word clouds, paired.
            Both have the same word-cloud shape, so the eye reads them
            as a single comparison: "this is the breadth of objects we
            see, this is the breadth of actions". */}
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard
            src={egoObjectUrl}
            alt="Word cloud of 800+ object categories observed in the egocentric subset"
            tag="02 · objects"
            title="150,000+ object instances across 800+ categories."
            caption="From `smartphone` and `keyboard` down to `tarot cards` and `airbrush`. Long-tail object exposure is what lets a policy generalize to the manipulation scenarios researchers actually study."
            aspectRatio="1024 / 512"
          />
          <ChartCard
            src={egoTaskUrl}
            alt="Word cloud of action and task labels covered in HumanNet"
            tag="03 · actions"
            title="720,000+ task instances spanning fine-grained actions."
            caption="Daily activities (`eating with chopsticks`, `folding laundry`), social interactions (`waving`, `clapping`), locomotion, and long-horizon assemblies — captured at clip granularity, not just video-level tags."
            aspectRatio="1024 / 512"
          />
        </div>

        {/* Row 3 — Distribution bars, paired.
            Same chart family (horizontal stacked-bar with category +
            sub-category), so the visual rhythm matches even though the
            two domains (scene vs motion) differ. */}
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard
            src={sceneBarUrl}
            alt="Horizontal bar chart breaking down scene coverage by sub-category"
            tag="04 · scene breakdown"
            title="Indoor heavy, but the tail matters."
            caption="`Home/other-indoor` (33.3%) and `kitchen` (17.4%) anchor the head. The 11% industrial / construction tail is where most existing open datasets fall off — HumanNet keeps it in."
            aspectRatio="1024 / 900"
          />
          <ChartCard
            src={exoMotionUrl}
            alt="Horizontal bar chart of motion categories and sub-categories"
            tag="05 · motion breakdown"
            title="Daily activity beyond sports highlights."
            caption="Daily activities lead at 53.6% (basic postures, object interaction, eating, fine-motor skills). Social leisure adds 39.1%. Sports and game-character actions stay below 1%, keeping the distribution focused on embodied behavior."
            aspectRatio="1024 / 600"
          />
        </div>

        {/* Row 4 — Motion taxonomy sunburst.
            This one has integrated text legends on the right, so the
            wide aspect already plates well at full width. */}
        <ChartCard
          src={motionSunburstUrl}
          alt="Three-level sunburst of the motion-category taxonomy with summary statistics"
          tag="06 · motion taxonomy"
          title="6 top-level / 56 mid-level / 828 leaf actions."
          caption="The full action taxonomy is committed to the repo and frozen for the Preview release. New categories land via versioned PRs so cross-version analysis stays comparable."
          aspectRatio="1024 / 600"
        />

        {/* Row 5 — Quality / duration dashboard.
            Six small panels — pose-score, motion-score, frame-count
            distributions plus per-category means and a metrics table.
            Looks like a research figure on purpose. */}
        <ChartCard
          src={motionDashboardUrl}
          alt="Dashboard of motion-quality and clip-duration distributions across pose score, motion score, frame count, and per-category means"
          tag="07 · quality & duration"
          title="Quality signals are first-class — not an afterthought."
          caption="Pose-score and motion-score distributions are released alongside the clips, so subset-selection and curriculum experiments can reproduce the same quality slices we used to evaluate baselines."
          aspectRatio="1024 / 620"
        />
      </div>
    </Section>
  );
}
