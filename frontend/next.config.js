/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    typedRoutes: false,
  },
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // ignoreBuildErrors: true,
  },
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  images: {
    domains: [],
  },
  env: {
    CUSTOM_KEY: process.env.CUSTOM_KEY,
  },
  // Proxy API requests to backend server
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
  // Increase timeout for long-running requests
  experimental: {
    proxyTimeout: 300000, // 5 minutes
  },
}

module.exports = nextConfig
