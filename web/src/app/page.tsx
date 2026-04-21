"use client";

import { useState } from "react";
import useSWR from "swr";
import dynamic from "next/dynamic";
import clsx from "clsx";

const MissionMap = dynamic(() => import("@/components/map/MissionMap").then((m) => m.MissionMap), {
  ssr: false,
  loading: () => <div className="empty">loading map…</div>
});

const fetcher = (url: string) => fetch(url).then((r) => r.json());

const SAMPLE_AOIS = [
  { name: "Coastal A",   minLatDeg: 32.6, maxLatDeg: 33.0, minLonDeg: -118.4, maxLonDeg: -117.9 },
  { name: "Mountain B",  minLatDeg: 39.2, maxLatDeg: 39.6, minLonDeg: -106.1, maxLonDeg: -105.7 },
  { name: "Desert C",    minLatDeg: 35.0, maxLatDeg: 35.4, minLonDeg: -116.0, maxLonDeg: -115.6 },
  { name: "Urban D",     minLatDeg: 40.7, maxLatDeg: 40.9, minLonDeg: -74.10, maxLonDeg: -73.90 },
  { name: "Strait E",    minLatDeg: 36.0, maxLatDeg: 36.4, minLonDeg:   5.6,  maxLonDeg:   6.0 }
];

const ALL_ANALYTICS = ["REGIME_DETECT", "CHANGE_POINT", "TASKING", "UNCERTAINTY", "ANOMALY"];

interface Mission {
  id: string;
  name: string;
  aoi: typeof SAMPLE_AOIS[0];
  priority: "ROUTINE" | "PRIORITY" | "IMMEDIATE" | "FLASH";
  state: string;
  analytics: string[];
}

export default function MissionsPage() {
  const { data: missions, mutate } = useSWR<Mission[]>("/api/missions", fetcher, { refreshInterval: 3000 });
  const { data: stats } = useSWR("/api/missions/stats", fetcher, { refreshInterval: 3000 });
  const [aoi, setAoi] = useState(SAMPLE_AOIS[0]);
  const [priority, setPriority] = useState<Mission["priority"]>("PRIORITY");
  const [name, setName] = useState("Sentinel Sweep");
  const [submitting, setSubmitting] = useState(false);

  const submit = async () => {
    setSubmitting(true);
    try {
      await fetch("/api/missions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name, aoi, priority,
          analytics: ALL_ANALYTICS,
          operator: "console-user"
        })
      });
      mutate();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="split">
      <div style={{ display: "flex", flexDirection: "column", gap: 12, minHeight: 0 }}>
        <div className="card">
          <div className="card__title">Mission queue</div>
          <div className="grid grid--4" style={{ marginTop: 8 }}>
            <Stat label="Total" v={stats?.total ?? 0} />
            <Stat label="Sched." v={stats?.scheduled ?? 0} />
            <Stat label="Run" v={stats?.running ?? 0} info />
            <Stat label="Done" v={stats?.completed ?? 0} ok />
          </div>
        </div>
        <div className="card">
          <div className="card__title">New mission</div>
          <label className="label">Name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
          <label className="label" style={{ marginTop: 8 }}>AOI</label>
          <select className="select" value={aoi.name}
                  onChange={(e) => setAoi(SAMPLE_AOIS.find((a) => a.name === e.target.value)!)}>
            {SAMPLE_AOIS.map((a) => <option key={a.name}>{a.name}</option>)}
          </select>
          <label className="label" style={{ marginTop: 8 }}>Priority</label>
          <select className="select" value={priority}
                  onChange={(e) => setPriority(e.target.value as Mission["priority"])}>
            <option>ROUTINE</option><option>PRIORITY</option>
            <option>IMMEDIATE</option><option>FLASH</option>
          </select>
          <button className="btn" style={{ marginTop: 12, width: "100%" }} disabled={submitting} onClick={submit}>
            {submitting ? "submitting…" : "task mission"}
          </button>
        </div>
        <div className="card scroll-y" style={{ flex: 1, minHeight: 0 }}>
          <div className="card__title">Active missions</div>
          {(!missions || missions.length === 0) && <div className="empty">no missions yet</div>}
          {missions?.map((m) => (
            <div key={m.id} style={{
              padding: "8px 10px", background: "var(--bg-2)", borderRadius: 6,
              marginBottom: 6, display: "flex", flexDirection: "column", gap: 4
            }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <strong style={{ fontSize: 13 }}>{m.name}</strong>
                <span className={clsx("tag", `tag--${m.priority.toLowerCase()}`)}>{m.priority}</span>
              </div>
              <div style={{ color: "var(--dim)", fontSize: 11 }}>
                {m.aoi.name} · {m.analytics.length} analytics
              </div>
              <span className={clsx("tag", `tag--state-${m.state.toLowerCase()}`)} style={{ alignSelf: "flex-start" }}>
                {m.state}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="map-panel">
        <MissionMap selectedAoi={aoi} missions={missions ?? []} />
      </div>
    </div>
  );
}

function Stat({ label, v, info, ok }: { label: string; v: number; info?: boolean; ok?: boolean }) {
  return (
    <div style={{
      background: "var(--bg-2)", padding: 8, borderRadius: 6, textAlign: "center",
      borderTop: info ? "2px solid var(--info)" : ok ? "2px solid var(--ok)" : "2px solid transparent"
    }}>
      <div style={{ fontSize: 11, color: "var(--dim)", textTransform: "uppercase", letterSpacing: 0.06 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 600 }}>{v}</div>
    </div>
  );
}
