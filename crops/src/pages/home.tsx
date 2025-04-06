import { Button } from "../components/ui/button";
import { Link } from "react-router-dom";
import { Sprout } from "lucide-react";
import { Card } from "../components/ui/card";

export function Home() {
  const features = [
    {
      title: "Accurate Analysis",
      description: "Get precise soil composition data using advanced testing methods",
      icon: "📊",
    },
    {
      title: "Crop Recommendations",
      description: "Receive tailored crop suggestions based on your soil profile",
      icon: "🌾",
    },
    {
      title: "Historical Tracking",
      description: "Monitor soil health changes over time with detailed records",
      icon: "📈",
    },
  ];

  return (
    <div className="container py-12 space-y-16">
      <div className="flex flex-col items-center text-center space-y-8">
        <div className="rounded-full bg-primary/10 p-4">
          <Sprout className="h-12 w-12 text-primary" />
        </div>
        <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl">
          Predict-O-Crop
        </h1>
        <p className="max-w-[700px] text-lg text-muted-foreground">
          Make informed decisions about your crops with our advanced soil analysis tools.
          Get detailed insights and recommendations based on your soil composition.
        </p>
        <div className="flex flex-wrap gap-4 justify-center">
          <Button asChild size="lg" className="h-12 px-8">
            <Link to="/soil-sampling">Start Soil Sampling</Link>
          </Button>
          <Button asChild size="lg" variant="outline" className="h-12 px-8">
            <Link to="/about">Learn More</Link>
          </Button>
        </div>
      </div>

      <div className="grid gap-8 md:grid-cols-3">
        {features.map((feature) => (
          <Card key={feature.title} className="p-6 text-center">
            <div className="text-4xl mb-4">{feature.icon}</div>
            <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
            <p className="text-muted-foreground">{feature.description}</p>
          </Card>
        ))}
      </div>

      <div className="bg-muted rounded-lg p-8 text-center">
        <h2 className="text-2xl font-bold mb-4">Trusted by Farmers Worldwide</h2>
        <p className="text-muted-foreground mb-6">
          Join thousands of farmers who make better farming decisions with our soil analysis system
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="flex flex-col items-center">
              <div className="text-3xl font-bold text-primary mb-2">
                {(i + 1) * 1000}+
              </div>
              <div className="text-sm text-muted-foreground">
                {i === 0 ? "Soil Samples" : i === 1 ? "Active Users" : i === 2 ? "Crop Varieties" : "Success Stories"}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}