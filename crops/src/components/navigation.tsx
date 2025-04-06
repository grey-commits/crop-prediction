import { Link, useLocation } from "react-router-dom";
import { cn } from "../lib/utils";
import { Button } from "./ui/button";

const navigation = [
  { name: "Home", href: "/" },
  { name: "Soil Sampling", href: "/soil-sampling" },
  { name: "Analysis", href: "/analysis" },
  { name: "FAQ", href: "/faq" },
  { name: "About", href: "/about" },
  { name: "Contact", href: "/contact" },
];

export function Navigation() {
  const location = useLocation();

  return (
    <nav className="flex items-center space-x-4 lg:space-x-6">
      {navigation.map((item) => (
        <Button
          key={item.href}
          variant="ghost"
          asChild
          className={cn(
            "text-sm font-medium transition-colors hover:text-primary",
            location.pathname === item.href
              ? "text-primary"
              : "text-muted-foreground"
          )}
        >
          <Link to={item.href}>{item.name}</Link>
        </Button>
      ))}
    </nav>
  );
}