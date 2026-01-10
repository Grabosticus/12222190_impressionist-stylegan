import LandingSection from './components/LandingSection';
import GenerateSection from './components/GenerateSection';
import AboutSection from './components/AboutSection';
import Footer from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen">
      <LandingSection />
      <GenerateSection />
      <AboutSection />
      <Footer />
    </div>
  );
}
