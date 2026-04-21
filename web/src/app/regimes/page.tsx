"use client";

import { useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";

interface RegimeRes {
  states: number[];
  posterior: number[][];
  log_likelihood: number;
  means: number[];
  variances: number[];
}

export default function RegimesPage() {
  const [series, setSeries] = useState<number[]>(generateSampleSeries());
  const [resp, setResp] = useState<RegimeRes | null>(null);
  const [busy, setBusy] = useState(false);
  const [k, setK] = useState(2);

  const run = async () => {
    setBusy(true);
    try {
      const r = await fetch("/api/quant/regime", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ series, n_states: k })
      });
      setResp(await r.json());
    } finally {
      setBusy(false);
    }
  };

  const data = series.map((y, t) => ({
    t,
    value: y,
    state: resp ? resp.states[t] : null,
    p0: resp ? resp.posterior[t][0] : null
  }));

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>HMM regime decoding</h2>
      <p style={{ color: "var(--dim)", maxWidth: 720 }}>
        Same Baum-Welch / Viterbi machinery I use for vol-regime detection on
        intraday returns - here it's running on a synthetic NDVI revisit
        series with a planted regime shift.
      </p>
      <div className="card" style={{ marginBottom: 16, display: "flex", gap: 16, alignItems: "end" }}>
        <div>
          <label className="label">States</label>
          <input className="input" type="number" min={2} max={6} value={k}
                 onChange={(e) => setK(Number(e.target.value))} style={{ width: 100 }} />
        </div>
        <button className="btn" disabled={busy} onClick={run}>
          {busy ? "fitting…" : "fit HMM"}
        </button>
        <button className="btn btn--ghost" onClick={() => setSeries(generateSampleSeries(Math.random()))}>
          new sample
        </button>
      </div>
      <div className="card" style={{ height: 360 }}>
        <ResponsiveContainer>
          <ComposedChart data={data}>
            <XAxis dataKey="t" tick={{ fill: "#94a3c4", fontSize: 11 }} />
            <YAxis tick={{ fill: "#94a3c4", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0a1020", border: "1px solid #243152" }} />
            <Area dataKey="p0" stroke="#7dd3fc" fill="rgba(125,211,252,0.15)" />
            <Line dataKey="value" stroke="#2dd4bf" dot={false} strokeWidth={1.5} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {resp && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card__title">Fitted regimes</div>
          <table style={{ width: "100%", color: "var(--dim)", fontSize: 13 }}>
            <thead><tr><th align="left">k</th><th align="left">mean</th><th align="left">variance</th></tr></thead>
            <tbody>
              {resp.means.map((m, i) => (
                <tr key={i}>
                  <td>{i}</td>
                  <td>{m.toFixed(3)}</td>
                  <td>{resp.variances[i].toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 8, color: "var(--dim)" }}>
            log-likelihood {resp.log_likelihood.toFixed(2)}
          </div>
        </div>
      )}
    </div>
  );
}

function generateSampleSeries(seed = 1): number[] {
  const out: number[] = [];
  const n = 240, shift = 120;
  let s = seed * 13;
  for (let i = 0; i < n; i++) {
    s = (s * 9301 + 49297) % 233280;
    const r = (s / 233280) - 0.5;
    const m = i < shift ? 0.55 : 0.30;
    out.push(m + 0.15 * Math.sin(i / 10) + 0.06 * r);
  }
  return out;
}
