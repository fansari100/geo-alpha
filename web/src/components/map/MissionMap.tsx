"use client";

import { MapContainer, TileLayer, Rectangle, Tooltip, useMap } from "react-leaflet";
import L from "leaflet";
import { useEffect } from "react";

interface AOI {
  name: string;
  minLatDeg: number;
  maxLatDeg: number;
  minLonDeg: number;
  maxLonDeg: number;
}
interface Mission {
  id: string;
  name: string;
  aoi: AOI;
  priority: "ROUTINE" | "PRIORITY" | "IMMEDIATE" | "FLASH";
  state: string;
}

const PRIO_COLOR: Record<Mission["priority"], string> = {
  ROUTINE: "#60a5fa",
  PRIORITY: "#2dd4bf",
  IMMEDIATE: "#fbbf24",
  FLASH: "#f87171"
};

function FitTo({ aoi }: { aoi: AOI }) {
  const map = useMap();
  useEffect(() => {
    const b = L.latLngBounds([aoi.minLatDeg, aoi.minLonDeg], [aoi.maxLatDeg, aoi.maxLonDeg]);
    map.fitBounds(b.pad(2.5));
  }, [aoi, map]);
  return null;
}

export function MissionMap({ selectedAoi, missions }: { selectedAoi: AOI; missions: Mission[] }) {
  const center: [number, number] = [
    (selectedAoi.minLatDeg + selectedAoi.maxLatDeg) / 2,
    (selectedAoi.minLonDeg + selectedAoi.maxLonDeg) / 2
  ];
  return (
    <MapContainer center={center} zoom={6} scrollWheelZoom style={{ height: "100%", width: "100%" }}>
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution="&copy; OpenStreetMap &copy; CARTO"
      />
      <FitTo aoi={selectedAoi} />
      <Rectangle
        bounds={[[selectedAoi.minLatDeg, selectedAoi.minLonDeg], [selectedAoi.maxLatDeg, selectedAoi.maxLonDeg]]}
        pathOptions={{ color: "#7dd3fc", weight: 2, dashArray: "4 4", fillOpacity: 0.05 }}
      >
        <Tooltip permanent direction="top">{selectedAoi.name}</Tooltip>
      </Rectangle>
      {missions.map((m) => (
        <Rectangle
          key={m.id}
          bounds={[[m.aoi.minLatDeg, m.aoi.minLonDeg], [m.aoi.maxLatDeg, m.aoi.maxLonDeg]]}
          pathOptions={{
            color: PRIO_COLOR[m.priority],
            weight: m.state === "RUNNING" ? 3 : 1.5,
            fillColor: PRIO_COLOR[m.priority],
            fillOpacity: m.state === "COMPLETED" ? 0.18 : 0.08
          }}
        >
          <Tooltip>
            <strong>{m.name}</strong><br />
            {m.aoi.name} · {m.priority}<br />
            {m.state}
          </Tooltip>
        </Rectangle>
      ))}
    </MapContainer>
  );
}
