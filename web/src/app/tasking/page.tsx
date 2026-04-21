"use client";

import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface Target {
  name: string;
  value: number;
  dwell_max: number;
  priority: "ROUTINE" | "PRIORITY" | "IMMEDIATE" | "FLASH";
}

const PRIO_COLOR: Record<Target["priority"], string> = {
  ROUTINE: "#60a5fa",
  PRIORITY: "#2dd4bf",
  IMMEDIATE: "#fbbf24",
  FLASH: "#f87171"
};

const DEFAULT_TARGETS: Target[] = [
  { name: "alpha", value: 12.0, dwell_max: 30, priority: "ROUTINE" },
  { name: "bravo", value: 9.0, dwell_max: 30, priority: "PRIORITY" },
  { name: "charlie", value: 7.0, dwell_max: 30, priority: "IMMEDIATE" },
  { name: "delta", value: 4.5, dwell_max: 30, priority: "FLASH" },
  { name: "echo", value: 6.0, dwell_max: 30, priority: "ROUTINE" }
];

export default function TaskingPage() {
  const [budget, setBudget] = useState(90);
  const [targets, setTargets] = useState<Target[]>(DEFAULT_TARGETS);
  const [resp, setResp] = useState<{ assignment: Record<string, number>; total_value: number; solver: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const solve = async () => {
    setBusy(true);
    try {
      const r = await fetch("/api/quant/tasking", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targets, total_budget_s: budget, risk_aversion: 0.5 })
      });
      setResp(await r.json());
    } finally {
      setBusy(false);
    }
  };

  const data = targets.map((t) => ({
    name: t.name,
    dwell: resp?.assignment?.[t.name] ?? 0,
    color: PRIO_COLOR[t.priority]
  }));

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Sensor tasking</h2>
      <p style={{ color: "var(--dim)", maxWidth: 720 }}>
        Same convex optimization that powers long-only constrained
        portfolio allocation - here it allocates dwell-time across
        candidate AOIs subject to per-priority budget caps.
      </p>
      <div className="grid grid--3" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="card__title">Total budget (s)</div>
          <input className="input" type="number" value={budget}
                 onChange={(e) => setBudget(Number(e.target.value))} />
        </div>
        <div className="card">
          <div className="card__title">Targets</div>
          <div className="card__value">{targets.length}</div>
        </div>
        <div className="card" style={{ display: "flex", alignItems: "end" }}>
          <button className="btn" disabled={busy} onClick={solve} style={{ width: "100%" }}>
            {busy ? "solving…" : "solve"}
          </button>
        </div>
      </div>
      <div className="card" style={{ height: 320 }}>
        <ResponsiveContainer>
          <BarChart data={data}>
            <XAxis dataKey="name" tick={{ fill: "#94a3c4", fontSize: 11 }} />
            <YAxis tick={{ fill: "#94a3c4", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0a1020", border: "1px solid #243152" }} />
            <Bar dataKey="dwell" radius={[4, 4, 0, 0]}>
              {data.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {resp && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card__title">Solver result</div>
          <div className="card__value">{resp.total_value.toFixed(2)}</div>
          <div className="card__hint">
            {resp.solver} · sum dwell = {Object.values(resp.assignment).reduce((s, v) => s + v, 0).toFixed(1)}s
          </div>
        </div>
      )}
    </div>
  );
}
