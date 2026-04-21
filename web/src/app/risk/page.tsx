"use client";

import { useState } from "react";

export default function RiskPage() {
  const [resp, setResp] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      const scores = Array.from({ length: 4096 }, () => Math.abs(rng()) * 1.5);
      const r = await fetch("/api/quant/anomaly", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scores, threshold_quantile: 0.95, target_far: 1e-4 })
      });
      setResp(await r.json());
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Risk &amp; uncertainty</h2>
      <p style={{ color: "var(--dim)", maxWidth: 720 }}>
        EVT / GPD-calibrated anomaly threshold + Monte Carlo uncertainty
        propagation through the atmospheric correction chain.  Same
        contract I use on the quant side - target a false-alarm rate,
        get back a calibrated threshold, with the GPD shape parameter
        and scale exposed for sanity-checking.
      </p>
      <div className="card" style={{ marginBottom: 16 }}>
        <button className="btn" onClick={run} disabled={busy}>
          {busy ? "fitting…" : "fit GPD on synthetic scores"}
        </button>
      </div>
      {resp && (
        <div className="grid grid--3">
          <div className="card">
            <div className="card__title">EVT threshold</div>
            <div className="card__value">{resp.threshold.toFixed(3)}</div>
            <div className="card__hint">target FAR 1e-4</div>
          </div>
          <div className="card">
            <div className="card__title">GPD shape ξ</div>
            <div className="card__value">{resp.xi.toFixed(3)}</div>
            <div className="card__hint">{resp.xi > 0 ? "heavy tail" : "bounded tail"}</div>
          </div>
          <div className="card">
            <div className="card__title">Alarms</div>
            <div className="card__value">{resp.n_alarms}</div>
            <div className="card__hint">σ {resp.sigma.toFixed(3)}</div>
          </div>
        </div>
      )}
    </div>
  );
}

let _seed = 1;
function rng() {
  _seed = (_seed * 9301 + 49297) % 233280;
  return _seed / 233280 - 0.5;
}
