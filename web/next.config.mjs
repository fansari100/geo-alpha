/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/missions/:path*",
        destination: `${process.env.SERVICE_URL ?? "http://localhost:8080"}/api/v1/missions/:path*`
      },
      {
        source: "/api/quant/:path*",
        destination: `${process.env.GATEWAY_URL ?? "http://localhost:8000"}/quant/:path*`
      },
      {
        source: "/api/health",
        destination: `${process.env.GATEWAY_URL ?? "http://localhost:8000"}/health`
      }
    ];
  }
};

export default nextConfig;
