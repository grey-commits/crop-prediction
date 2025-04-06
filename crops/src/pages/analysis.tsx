import { Card } from "../components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const nutrientData = [
  { name: "Nitrogen (N)", value: 45 },
  { name: "Phosphorus (P)", value: 30 },
  { name: "Potassium (K)", value: 25 },
];

const moistureData = [
  { month: "Jan", value: 65 },
  { month: "Feb", value: 59 },
  { month: "Mar", value: 80 },
  { month: "Apr", value: 81 },
  { month: "May", value: 56 },
  { month: "Jun", value: 55 },
];

const COLORS = ["hsl(var(--chart-1))", "hsl(var(--chart-2))", "hsl(var(--chart-3))"];

export function Analysis() {
  return (
    <div className="container p-12">
      <h1 className="text-3xl font-bold mb-6">Soil Analysis</h1>
      <div className="grid gap-6 md:grid-cols-2">
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Nutrient Distribution</h2>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={nutrientData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {nutrientData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Moisture Levels</h2>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={moistureData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Moisture %" fill="hsl(var(--chart-1))" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
}