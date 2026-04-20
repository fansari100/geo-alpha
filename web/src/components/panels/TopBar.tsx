"use client";

import useSWR from "swr";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

export function TopBar() {
  const { data } = useSWR("/api/health", fetcher, { refreshInterval: 5_000 });
  const ok = data?.status === "ok";
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand__mark">α</div>
        <div>
          geo-alpha
          <div className="brand__sub">quantitative geospatial intelligence</div>
        </div>
      </div>
      <div className="status">
        <span className="pill">
          <span className={`dot ${ok ? "dot--ok" : "dot--err"}`} />
          gateway {ok ? "online" : "offline"}
        </span>
        <span className="pill">
          <span className="dot dot--ok" />
          v{data?.version ?? "0.1.0"}
        </span>
      </div>
    </header>
  );
}
