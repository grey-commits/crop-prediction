import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../components/ui/accordion";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Link } from "react-router-dom";

export function FAQ() {
  const faqs = [
    {
      question: "What is soil sampling and why is it important?",
      answer:
        "Soil sampling is the process of systematically collecting soil samples from different locations and depths within a specific area to assess its physical, chemical, and biological properties, such as nutrient levels, pH, texture, and organic matter content, in order to inform agricultural practices, land management, or environmental studies.",
    },
    {
      question: "How often should I conduct soil sampling?",
      answer:
        "We recommend conducting soil sampling at least once every 2-3 years. However, for intensive farming or when growing high-value crops, annual sampling might be beneficial. The frequency can also depend on soil type, crop rotation, and management practices.",
    },
    {
      question: "What's the best time to collect soil samples?",
      answer:
        "The best time to collect soil samples is typically during the fall or early spring when the soil is not overly wet or frozen. Fall sampling allows for nutrient adjustments before planting, while early spring sampling helps to plan for the upcoming growing season. Avoid sampling when the soil is excessively wet, dry, or frozen, as these conditions can lead to inaccurate results.",
    },
    {
      question: "How deep should I sample the soil?",
      answer:
        "For most agricultural purposes, samples should be taken from the top 6-8 inches of soil, as this is the primary root zone for most crops. However, for deep-rooted crops or specific analyses, deeper sampling might be necessary.",
    },
    {
      question: "How many samples should I take per field?",
      answer:
        "The number of samples depends on field size and variability. A general rule is to take 15-20 subsamples per 20 acres for uniform fields. For fields with varying soil types or topography, more samples might be needed.",
    },
    {
      question: "What nutrients does the analysis measure?",
      answer:
        "Our standard analysis includes major nutrients (N, P, K), secondary nutrients (Ca, Mg, S), micronutrients (Zn, Mn, Cu, B), pH, organic matter content, and cation exchange capacity (CEC).",
    },
    {
      question: "How long does it take to get results?",
      answer:
        "Standard analysis results are typically available within 5-7 business days after we receive your samples. Rush service is available for an additional fee, with results in 2-3 business days.",
    },
    {
      question: "How accurate are the results?",
      answer:
        "Our laboratory uses state-of-the-art equipment and follows strict quality control procedures. Results are typically accurate within ±5% for most parameters, meeting or exceeding industry standards.",
    },
  ];

  const categories = [
    "Sampling Methods",
    "Equipment",
    "Analysis",
    "Results",
    "Recommendations",
    "Pricing",
  ];

  return (
    <div className="container py-12">
      <div className="text-center max-w-2xl mx-auto mb-12">
        <h1 className="text-4xl font-bold mb-4">Frequently Asked Questions</h1>
        <p className="text-lg text-muted-foreground">
          Find answers to common questions about our soil sampling system and
          analysis process.
        </p>
      </div>

      <div className="grid md:grid-cols-4 gap-8">
        {/* FAQ Accordion */}
        <div className="md:col-span-3 text-slate-900">
          <Accordion
            type="single"
            collapsible
            className="space-y-4 px-12 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700"
          >
            {faqs.map((faq, index) => (
              <AccordionItem
                key={index}
                value={`item-${index}`}
                className="rounded-lg border border-gray-200 dark:border-gray-700"
              >
                <AccordionTrigger className="text-left px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-t-lg">
                  {faq.question}
                </AccordionTrigger>
                <AccordionContent className="px-4 py-2 bg-gray-50 dark:bg-gray-800 rounded-b-lg">
                  {faq.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>

        {/* Categories and Support */}
        <div className="space-y-6">
          {/* Categories */}
          <Card className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold mb-4 text-gray-800 dark:text-gray-100">
              Browse by Category
            </h2>
            <div className="space-y-2">
              {categories.map((category) => (
                <Button
                  key={category}
                  variant="ghost"
                  className="w-full justify-start bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-100 rounded-md"
                >
                  {category}
                </Button>
              ))}
            </div>
          </Card>

          {/* Support */}
          <Card className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
            <h2 className="font-semibold mb-4 text-gray-800 dark:text-gray-100">
              Still Have Questions?
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Can't find the answer you're looking for? Reach out to our support
              team.
            </p>
            <Button
              asChild
              className="w-full bg-blue-500 hover:bg-blue-600 text-white rounded-md"
            >
              <Link to="/contact">Contact Support</Link>
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}