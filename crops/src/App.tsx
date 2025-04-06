import { ThemeProvider } from './components/theme-provider';
import { Routes } from './components/routes';
import { Toaster } from './components/ui/sonner';

function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="vite-ui-theme">
      <main className="min-h-screen bg-background ">
        <Routes />
        <Toaster />
      </main>
    </ThemeProvider>
  );
}

export default App;