import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/panels/Sidebar";
import { TopBar } from "@/components/panels/TopBar";

export const metadata: Metadata = {
  title: "geo-alpha · operator console",
  description: "Quantitative methods for geospatial intelligence."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <TopBar />
          <div className="body">
            <Sidebar />
            <main className="content">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
