import { Outlet } from "react-router-dom";
import { Navigation } from "./navigation";
import { ThemeToggle } from "./theme-toggle";

export function Layout() {
  return (
    <div className="w-screen flex-col content-center">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center">
          <Navigation />
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="border-t">
        <div className="container py-6 text-center text-sm text-muted-foreground">
          © 2025 Soil Sampling System. All rights reserved.
        </div>
      </footer>
    </div>
  );
}