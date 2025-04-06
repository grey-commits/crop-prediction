import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Layout } from "./layout";
import { Home } from '../pages/home';
import { About } from "../pages/about";
import { Contact } from "../pages/contact";
import { FAQ } from "../pages/faq";
import { SoilSamplingForm } from "../pages/soil-sampling-form";
import { Results } from "../pages/results";
import { Analysis } from "../pages/analysis";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      { path: "about", element: <About /> },
      { path: "contact", element: <Contact /> },
      { path: "faq", element: <FAQ /> },
      { path: "soil-sampling", element: <SoilSamplingForm /> },
      { path: "results", element: <Results /> },
      { path: "analysis", element: <Analysis /> },
    ],
  },
]);

export function Routes() {
  return <RouterProvider router={router} />;
}