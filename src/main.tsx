import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import App from "./App";
import { AdminApp } from "./admin/AdminApp";
import { ConfigProvider } from "./lib/useConfig";
import { ThemeProvider } from "./lib/theme";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <ConfigProvider>
          <Routes>
            <Route path="/" element={<App />} />
            <Route path="/admin/*" element={<AdminApp />} />
          </Routes>
        </ConfigProvider>
      </ThemeProvider>
    </BrowserRouter>
  </StrictMode>,
);
