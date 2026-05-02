import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { adminApi, getToken, setToken } from "./api";
import { AdminLogin } from "./AdminLogin";
import { AdminDashboard } from "./AdminDashboard";

/**
 * Admin SPA mounted at /admin/*. Two routes:
 *   /admin/login      → username + password form
 *   /admin            → dashboard (config editor)
 *
 * The router pushes unauthenticated users to /admin/login on first
 * render. After a successful login the dashboard renders inline; the
 * token lives in localStorage so a refresh keeps you in.
 */
export function AdminApp() {
  return (
    <div className="min-h-dvh bg-nebula-base text-nebula-on">
      <Routes>
        <Route index element={<AuthGate><AdminDashboard /></AuthGate>} />
        <Route path="login" element={<AdminLogin />} />
        <Route path="*" element={<Navigate to="/admin" replace />} />
      </Routes>
    </div>
  );
}

function AuthGate({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<"checking" | "ok" | "deny">("checking");
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const token = getToken();
      if (!token) {
        if (!cancelled) navigate("/admin/login", { replace: true });
        return;
      }
      try {
        const me = await adminApi.me();
        if (cancelled) return;
        if (me.authenticated) setState("ok");
        else {
          setToken(null);
          navigate("/admin/login", { replace: true });
        }
      } catch {
        if (!cancelled) {
          setToken(null);
          navigate("/admin/login", { replace: true });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [navigate]);

  if (state !== "ok") {
    return (
      <div className="grid min-h-dvh place-items-center">
        <div className="font-mono text-xs uppercase tracking-[0.22em] text-nebula-on-dim">
          → Authenticating …
        </div>
      </div>
    );
  }
  return <>{children}</>;
}
