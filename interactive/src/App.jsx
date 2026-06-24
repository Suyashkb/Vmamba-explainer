import { AnimatePresence, motion } from "framer-motion";
import useZoomLevel from "./hooks/useZoomLevel";
import useReducedMotion from "./hooks/useReducedMotion";
import NavBar from "./components/shared/NavBar";
import InfoPanel from "./components/shared/InfoPanel";
import L0_MacroPipeline from "./components/zoom-levels/L0_MacroPipeline";
import L1_VSSBlock from "./components/zoom-levels/L1_VSSBlock";
import L2_SS2D from "./components/zoom-levels/L2_SS2D";
import L3_SSMCell from "./components/zoom-levels/L3_SSMCell";
import "./styles/app.css";

const LEVEL_COMPONENTS = {
  L0: L0_MacroPipeline,
  L1: L1_VSSBlock,
  L2: L2_SS2D,
  L3: L3_SSMCell,
};

function makeVariants(reduced) {
  if (reduced) {
    return {
      initial: { opacity: 0 },
      animate: { opacity: 1, transition: { duration: 0 } },
      exit:    { opacity: 0, transition: { duration: 0 } },
    };
  }
  return {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0,  transition: { duration: 0.22, ease: "easeOut" } },
    exit:    { opacity: 0, y: -8, transition: { duration: 0.15, ease: "easeIn"  } },
  };
}

function App() {
  const { level } = useZoomLevel();
  const reduced = useReducedMotion();
  const LevelComponent = LEVEL_COMPONENTS[level];
  const variants = makeVariants(reduced);

  return (
    <div className={`app-shell${reduced ? " reduced-motion-static" : ""}`}>
      <NavBar />

      <main className="main-canvas" id="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={level}
            variants={variants}
            initial="initial"
            animate="animate"
            exit="exit"
            style={{ position: "absolute", inset: 0 }}
          >
            <LevelComponent />
          </motion.div>
        </AnimatePresence>
      </main>

      <InfoPanel />
    </div>
  );
}

export default App;
