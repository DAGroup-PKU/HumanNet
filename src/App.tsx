import { Navbar } from "./components/Navbar";
import { Hero } from "./components/Hero";
import { DatasetProfile } from "./components/DatasetProfile";
import { DataScale } from "./components/DataScale";
import { PerspectiveExplorer } from "./components/PerspectiveExplorer";
import { DataGallery } from "./components/DataGallery";
import { Roadmap } from "./components/Roadmap";
import { Members } from "./components/Members";
import { Waitlist } from "./components/Waitlist";
import { Footer } from "./components/Footer";

const SHOW_DATA_SCALE = false;

export default function App() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <DatasetProfile />
        {SHOW_DATA_SCALE ? (
          <>
            <div className="section-divider" aria-hidden="true" />
            <DataScale />
          </>
        ) : null}
        <div className="section-divider" aria-hidden="true" />
        <PerspectiveExplorer />
        <div className="section-divider" aria-hidden="true" />
        <DataGallery />
        <div className="section-divider" aria-hidden="true" />
        <Roadmap />
        <div className="section-divider" aria-hidden="true" />
        <Members />
        <div className="section-divider" aria-hidden="true" />
        <Waitlist />
      </main>
      <Footer />
    </>
  );
}
