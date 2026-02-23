/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow images from your API
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
      },
    ],
  },
  // Proxy API calls to FastAPI in development
  async rewrites() {
    return [
      {
        source: '/auth/:path*',
        destination: 'http://localhost:8000/auth/:path*',
      },
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      { source: '/images/:path*', destination: 'http://127.0.0.1:8000/images/:path*' },
      { source: '/api/:path*', destination: 'http://127.0.0.1:8000/api/:path*' },
      { source: '/convos', destination: 'http://127.0.0.1:8000/convos' },
      { source: '/bulk/:path*', destination: 'http://127.0.0.1:8000/bulk/:path*' },

    ];
  },
};

export default nextConfig;
