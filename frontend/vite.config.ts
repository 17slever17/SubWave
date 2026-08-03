import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const backendPort = process.env.REALTIME_WEBUI_BACKEND_PORT || env.REALTIME_WEBUI_BACKEND_PORT || '7860'

  return {
    plugins: [react(), tailwindcss()],
    define: {
      'import.meta.env.VITE_REALTIME_BACKEND_PORT': JSON.stringify(backendPort),
    },
    test: {
      environment: 'jsdom',
      setupFiles: './src/test/setup.ts',
      globals: true,
      restoreMocks: true,
    },
    server: {
      proxy: {
        '/api': `http://127.0.0.1:${backendPort}`,
      },
    },
    build: {
      outDir: '../static',
      emptyOutDir: true,
    },
  }
})
