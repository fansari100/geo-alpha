"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const ITEMS = [
  { href: "/", label: "Missions" },
  { href: "/regimes", label: "Regimes" },
  { href: "/tasking", label: "Tasking" },
  { href: "/risk", label: "Risk" }
];

export function Sidebar() {
  const path = usePathname();
  return (
    <nav className="sidebar">
      {ITEMS.map((it) => {
        const active = path === it.href;
        return (
          <Link key={it.href} href={it.href} className={clsx("nav-item", active && "nav-item--active")}>
            {it.label}
          </Link>
        );
      })}
    </nav>
  );
}
