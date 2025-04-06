import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Textarea } from "../components/ui/textarea";
import { Mail, Phone, MapPin } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

export function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Simulating form submission
    toast.success("Message sent successfully! We'll get back to you soon.");
    setFormData({ name: "", email: "", subject: "", message: "" });
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const contactInfo = [
    {
      icon: <Mail className="h-6 w-6" />,
      title: "Email",
      value: "fategray2004@gmail.com",
      link: "mailto:support@soilsampling.com",
    },
    {
      icon: <Phone className="h-6 w-6" />,
      title: "Phone",
      value: "6386297791",
      link: "tel:+916386297791",
    },
    {
      icon: <MapPin className="h-6 w-6" />,
      title: "Address",
      value: "132 himadri boys hostel",
      link: "https://maps.google.com",
    },
  ];

  return (
    <div className="container py-12">
      <div className="text-center max-w-2xl mx-auto mb-12">
        <h1 className="text-4xl font-bold mb-4">Contact Us</h1>
        <p className="text-lg text-muted-foreground">
          Have questions about our soil sampling system? We're here to help!
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <Card className="p-6">
          <h2 className="text-2xl font-semibold mb-6">Send us a Message</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="subject">Subject</Label>
              <Input
                id="subject"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="message">Message</Label>
              <Textarea
                id="message"
                name="message"
                value={formData.message}
                onChange={handleChange}
                required
                className="min-h-[150px]"
              />
            </div>
            <Button type="submit" className="w-full">
              Send Message
            </Button>
          </form>
        </Card>

        <div className="space-y-8">
          <div>
            <h2 className="text-2xl font-semibold mb-6">Contact Information</h2>
            <div className="space-y-4">
              {contactInfo.map((info) => (
                <a
                  key={info.title}
                  href={info.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-4 p-4 rounded-lg hover:bg-muted transition-colors"
                >
                  <div className="text-primary">{info.icon}</div>
                  <div>
                    <h3 className="font-medium">{info.title}</h3>
                    <p className="text-muted-foreground">{info.value}</p>
                  </div>
                </a>
              ))}
            </div>
          </div>

          
        </div>
      </div>
    </div>
  );
}