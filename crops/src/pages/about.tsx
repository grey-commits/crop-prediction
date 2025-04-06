import { Card } from "../components/ui/card";
import { Leaf, Users, Trophy, Target } from "lucide-react";

export function About() {
  const team = [
    {
      name: "Mridul Sharma",
      role: "ML Expert",
      image: "./ms2.jpeg",
    },
    {
      name: "Akshat Mani Tripathi",
      role: "Frontend Developer",
      image: "./akshat2.jpeg",
    },
    {
      name: "Shagun Rana",
      role: "Data Analyst",
      image: "./sr.jpg",
    },
    {
      name: "Tanishka Khandelwal",
      role: "UI Designer",
      image: "./tk.jpeg",
    }
  ];

  const values = [
    {
      icon: <Leaf className="h-8 w-8" />,
      title: "Sustainability",
      description: "Promoting environmentally conscious farming practices",
    },
    {
      icon: <Users className="h-8 w-8" />,
      title: "Community",
      description: "Building a network of knowledgeable farmers",
    },
    {
      icon: <Trophy className="h-8 w-8" />,
      title: "Excellence",
      description: "Delivering accurate and reliable soil analysis",
    },
    {
      icon: <Target className="h-8 w-8" />,
      title: "Innovation",
      description: "Continuously improving our technology",
    },
  ];

  return (
    <div className="container py-12 space-y-16">
      <div className="text-center max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-6">About Our Mission</h1>
        <p className="text-lg text-muted-foreground">
          We're dedicated to revolutionizing agriculture through precise soil analysis
          and data-driven recommendations. Our mission is to help farmers maximize
          their yields while maintaining sustainable practices.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <div className="pl-6">
          <h2 className="text-2xl font-semibold mb-4">Our Story</h2>
          <p className="text-muted-foreground mb-4">
          This project started as a student initiative aimed at solving real-world agricultural challenges through technology. What began as a learning experience has grown into a fully functional platform that provides precise soil analysis and crop recommendations. Our journey has been fueled by curiosity, research, and a commitment to making an impact in the agricultural sector. 
          </p>
          <p className="text-muted-foreground">
            Today, we continue to innovate and improve our services, combining
            cutting-edge technology with agricultural expertise to provide the most
            accurate soil analysis possible.
          </p>
        </div>
        <Card className="p-6">
  <h3 className="text-xl font-semibold mb-4">Key Achievements</h3>
  <ul className="space-y-4">
    <li className="flex items-start gap-4">
      <div className="rounded-full bg-primary/10 p-2">
        <Trophy className="h-4 w-4 text-primary" />
      </div>
      <div>
        <h4 className="font-medium">A Step Towards Smarter Farming</h4>
        <p className="text-sm text-muted-foreground">
          Exploring innovative solutions to make agriculture more efficient and data-driven.
        </p>
      </div>
    </li>
    <li className="flex items-start gap-4">
      <div className="rounded-full bg-primary/10 p-2">
        <Users className="h-4 w-4 text-primary" />
      </div>
      <div>
        <h4 className="font-medium">Built with Passion & Purpose</h4>
        <p className="text-sm text-muted-foreground">
          Created by students eager to bring technology into farming for a better future.
        </p>
      </div>
    </li>
  </ul>
</Card>
      </div>

      <div>
        <h2 className="text-2xl font-semibold mb-8 text-center">Our Values</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {values.map((value) => (
            <Card key={value.title} className="p-6 text-center">
              <div className="mb-4 text-primary flex justify-center">
                {value.icon}
              </div>
              <h3 className="font-semibold mb-2">{value.title}</h3>
              <p className="text-sm text-muted-foreground">
                {value.description}
              </p>
            </Card>
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-semibold mb-8 text-center">Our Team</h2>
        <div className="grid md:grid-cols-4 gap-8 h-auto">
          {team.map((member) => (
            <Card key={member.name} className="overflow-hidden flex flex-col justify-between ">
              <img
                src={member.image}
                alt={member.name}
                className="w-full h-96 p-2 object-cover"
              />
              <div className="p-4 text-center">
                <h3 className="font-semibold">{member.name}</h3>
                <p className="text-sm text-muted-foreground">{member.role}</p>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}