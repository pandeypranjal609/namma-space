import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { spawn } from 'node:child_process';

function pythonAiBackendPlugin() {
  let pyProc = null;
  return {
    name: 'namma-space-python-ai-backend',
    configureServer() {
      if (pyProc) return;
      console.log('[Vite] Starting Namma Space Python AI Backend (pipeline/server.py)...');
      pyProc = spawn('python', ['pipeline/server.py'], {
        stdio: 'inherit',
        shell: true
      });
      pyProc.on('error', (err) => {
        console.warn('[Vite] Python backend could not auto-start:', err.message);
      });
      process.on('exit', () => {
        if (pyProc) pyProc.kill();
      });
    }
  };
}

export default defineConfig({
  plugins: [basicSsl(), pythonAiBackendPlugin()],
  server: {
    port: 5173,
    host: true,
    https: true,
    open: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  },
  build: {
    target: 'esnext',
    outDir: 'dist'
  }
});
