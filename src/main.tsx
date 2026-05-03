import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import App from "./App";
import { ConfigProvider } from "./lib/useConfig";
import { ThemeProvider } from "./lib/theme";
import "./index.css";

const STATIC_SITE = import.meta.env.VITE_STATIC_SITE === "true";
const AdminApp = STATIC_SITE
  ? null
  : lazy(() =>
      import("./admin/AdminApp").then((mod) => ({ default: mod.AdminApp })),
    );

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <ThemeProvider>
        <ConfigProvider>
          <Routes>
            <Route path="/" element={<App />} />
            {!STATIC_SITE && AdminApp ? (
              <Route
                path="/admin/*"
                element={
                  <Suspense fallback={null}>
                    <AdminApp />
                  </Suspense>
                }
              />
            ) : null}
          </Routes>
        </ConfigProvider>
      </ThemeProvider>
    </BrowserRouter>
  </StrictMode>,
);
