const BASE_PATH = import.meta.env.BASE_URL.replace(/\/$/, "");

export function publicPath(path: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${BASE_PATH}${normalized}`;
}
