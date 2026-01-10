import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sparkles, Loader2, Download, RefreshCw, Clock, Palette as PaletteIcon } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

export default function GenerateSection() {
  const messages = [
    "I have been trained on works of over 100 artists!",
    "I have been optimized to generate artworks of a particular genre!",
    "My training took over 24 hours of GPU time!",
    "I am primarily a first-generation StyleGAN, but use 2nd generation techniques too!"
  ]
  const [message, setMessage] = useState(messages[0]);

  useEffect(() => {
      const pick = () => setMessage(messages[Math.floor(Math.random() * messages.length)]);
      pick();
      const id = setInterval(pick, 2500);
      return () => clearInterval(id);
  }, []);

  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationCount, setGenerationCount] = useState(0);
  const [isUpsampling, setIsUpsampling] = useState(false);
  const [imageResolution, setImageResolution] = useState<string | null>(null);

  const generateImage = async () => {
    setIsGenerating(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/generate", {method: "POST"});

      if (!res.ok)
        throw new Error(`ERROR during generate image request: ${res.status} ${res.statusText}`)

      const blob = await res.blob()
      const imageUrl = URL.createObjectURL(blob);

      setGeneratedImage(imageUrl);
      setGenerationCount(prev => prev + 1);
    } catch (err) {
      console.error(err)
    } finally {
      setIsGenerating(false);
      setImageResolution("64");
    }
  };

  const upsampleImage = async() => {
    if (!generatedImage || isUpsampling) return;
    setIsUpsampling(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/upsample", {method: "POST"});

      if (!res.ok)
        throw new Error(`ERROR during upsample image request: ${res.status} ${res.statusText}`)

      const blob = await res.blob()
      const imageUrl = URL.createObjectURL(blob);

      setGeneratedImage(imageUrl)
    } catch (err) {
      console.error(err)
    } finally {
      setIsUpsampling(false);
      setImageResolution(imageResolution * 4)
    }
  }

  return (
    <section id="generate-section" className="min-h-screen relative overflow-hidden bg-gradient-to-br from-amber-50 via-rose-50 to-purple-50">
      {/* Artistic Background Pattern */}
      <div className="absolute inset-0 opacity-40">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 25% 25%, rgba(251,191,36,0.15) 0%, transparent 50%),
                           radial-gradient(circle at 75% 75%, rgba(244,114,182,0.15) 0%, transparent 50%),
                           radial-gradient(circle at 50% 50%, rgba(168,85,247,0.1) 0%, transparent 50%)`,
        }} />
      </div>

      {/* Floating Brushstroke Effects */}
      <motion.div
        className="absolute top-20 left-10 size-96 bg-gradient-to-br from-amber-200/30 to-orange-200/20 rounded-full blur-3xl"
        animate={{
          x: [0, 50, 0],
          y: [0, 30, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute bottom-20 right-10 size-[30rem] bg-gradient-to-tl from-purple-200/30 to-blue-200/20 rounded-full blur-3xl"
        animate={{
          x: [0, -30, 0],
          y: [0, -50, 0],
          scale: [1, 1.15, 1],
        }}
        transition={{ duration: 25, repeat: Infinity, ease: "easeInOut" }}
      />

      <div className="relative z-10 py-24 px-4 max-w-6xl mx-auto">
        {/* Header Section */}
        <motion.div 
          className="text-center mb-20"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <motion.div 
            className="inline-flex items-center gap-2 mb-6 px-5 py-2.5 rounded-full bg-white/70 backdrop-blur-xl border border-purple-200/50 shadow-lg"
            whileHover={{ scale: 1.05 }}
          >
            <PaletteIcon className="size-4 text-purple-600" />
            <span className="text-sm tracking-wider text-purple-900">IMPRESSIONIST STYLEGAN</span>
          </motion.div>
          
          <h2 className="mb-6 text-5xl md:text-6xl lg:text-7xl bg-gradient-to-r from-amber-600 via-rose-600 to-purple-700 bg-clip-text text-transparent leading-tight">
            Generate Your Own Artwork
          </h2>
          
          <p className="text-lg md:text-xl text-gray-700 max-w-2xl mx-auto leading-relaxed font-light">
            By clicking the button, a deep neural network architecture trained on 23,328 impressionist paintings will create
            a new artwork for you
          </p>
        </motion.div>

        {/* Generation Button */}
        <motion.div
          className="flex justify-center mb-16"
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <motion.button
            onClick={generateImage}
            disabled={isGenerating}
            className="group relative bg-gradient-to-r from-amber-500 via-rose-500 to-purple-600 text-white px-14 py-6 rounded-2xl transition-all disabled:opacity-60 disabled:cursor-not-allowed overflow-hidden shadow-2xl"
            whileHover={{ scale: 1.05, boxShadow: "0 30px 60px -15px rgba(168, 85, 247, 0.4)" }}
            whileTap={{ scale: 0.95 }}
          >
            {/* Animated Background */}
            <motion.div 
              className="absolute inset-0 bg-gradient-to-r from-amber-400 via-rose-400 to-purple-500"
              initial={{ opacity: 0 }}
              whileHover={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            />
            
            {/* Button Content */}
            <span className="relative z-10 flex items-center gap-3 text-lg">
              {isGenerating ? (
                <>
                  <Loader2 className="size-6 animate-spin" />
                  <span className="tracking-wide">Painting Your Vision...</span>
                </>
              ) : (
                <>
                  <Sparkles className="size-6" />
                  <span className="tracking-wide">Generate Artwork</span>
                </>
              )}
            </span>
            
            {/* Shimmer Effect */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
              initial={{ x: '-100%' }}
              animate={{ x: '100%' }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 1 }}
            />
          </motion.button>
        </motion.div>

        {/* Generated Image Display */}
        <AnimatePresence mode="wait">
          {generatedImage && (
            <motion.div
              key={generationCount}
              initial={{ opacity: 0, y: 60, scale: 0.85 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.7, type: "spring", bounce: 0.3 }}
              className="max-w-3xl mx-auto"
            >
              {/* Canvas Frame Effect */}
              <div className="relative p-8 bg-gradient-to-br from-amber-100 via-white to-purple-100 rounded-3xl shadow-2xl">
                {/* Inner Shadow for Depth */}
                <div className="absolute inset-8 rounded-2xl shadow-inner pointer-events-none" />
                
                {/* Decorative Corner Flourishes */}
                <div className="absolute top-4 left-4 size-16 border-t-4 border-l-4 border-amber-400/40 rounded-tl-2xl" />
                <div className="absolute top-4 right-4 size-16 border-t-4 border-r-4 border-rose-400/40 rounded-tr-2xl" />
                <div className="absolute bottom-4 left-4 size-16 border-b-4 border-l-4 border-purple-400/40 rounded-bl-2xl" />
                <div className="absolute bottom-4 right-4 size-16 border-b-4 border-r-4 border-blue-400/40 rounded-br-2xl" />
                
                {/* Main Image Container */}
                <div className="relative overflow-hidden rounded-2xl shadow-2xl bg-white">
                  <motion.div
                    initial={{ scale: 1.1 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.6 }}
                    className="w-full h-full"
                  >
                    <ImageWithFallback
                      src={generatedImage}
                      alt="AI-generated impressionist artwork"
                      className="w-full h-full object-cover"
                    />
                  </motion.div>
                  
                  {/* Subtle Vignette Effect */}
                  <div className="absolute inset-0 shadow-[inset_0_0_100px_rgba(0,0,0,0.1)] pointer-events-none" />
                </div>

                {/* Image Info Bar */}
                <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4 px-2">
                  <div className="flex items-center gap-4">
                    <motion.div 
                      className="flex items-center gap-2 text-sm text-gray-700"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                      <div className="size-2.5 rounded-full bg-gradient-to-r from-green-400 to-emerald-500 animate-pulse shadow-lg shadow-green-500/50" />
                      <span className="font-medium">Impressionist Style</span>
                    </motion.div>
                    <div className="h-4 w-px bg-gray-300" />
                    <motion.div 
                      className="flex items-center gap-2 text-sm text-gray-600"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.4 }}
                    >
                      <Clock className="size-3.5" />
                      <span>Generated just now</span>
                    </motion.div>
                  </div>
                  
                  <motion.div 
                    className="flex gap-3"
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                  >
                    <motion.button
                      onClick={generateImage}
                      className="px-5 py-2.5 rounded-xl bg-white hover:bg-gradient-to-br hover:from-purple-50 hover:to-purple-100 border-2 border-purple-200 text-purple-900 transition-all flex items-center gap-2 shadow-md hover:shadow-lg"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      title="Generate new artwork"
                    >
                      <RefreshCw className="size-4" />
                      <span className="hidden sm:inline">New Creation</span>
                    </motion.button>
                    <motion.button
                      onClick={upsampleImage}
                      className="px-5 py-2.5 rounded-xl bg-white hover:bg-gradient-to-br hover:from-purple-50 hover:to-purple-100 border-2 border-purple-200 text-purple-900 transition-all flex items-center gap-2 shadow-md hover:shadow-lg"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      title="Upsample your image"
                    >
                      <Sparkles className="size-4"/>
                      <span className="hidden sm:inline">{isUpsampling ? "Upsampling..." : "Upsample Image"}</span>
                    </motion.button>
                    <motion.a
                      href={generatedImage}
                      download="impressionist-artwork.jpg"
                      className="px-5 py-2.5 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 hover:from-amber-400 hover:to-orange-500 text-white transition-all flex items-center gap-2 shadow-md hover:shadow-xl"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      title="Download your artwork"
                    >
                      <Download className="size-4" />
                      <span className="hidden sm:inline">Save Art</span>
                    </motion.a>
                  </motion.div>
                </div>
              </div>

              {/* Artwork Details Cards */}
              <motion.div 
                className="grid md:grid-cols-3 gap-6 mt-10"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                {[
                  { icon: '🎨', label: 'Artistic Style', value: 'Impressionist Landscapes', gradient: 'from-amber-400 to-orange-500' },
                  { icon: '✨', label: 'Inspired By', value: 'Over 100 artists', gradient: 'from-rose-400 to-pink-500' },
                  { icon: '🖼️', label: 'Resolution', value: `${imageResolution} × ${imageResolution} pixels`, gradient: 'from-purple-400 to-indigo-500' },
                ].map((item, index) => (
                  <motion.div
                    key={item.label}
                    className="relative bg-white/80 backdrop-blur-sm rounded-2xl p-6 border-2 border-white shadow-lg hover:shadow-xl transition-all group overflow-hidden"
                    whileHover={{ y: -5 }}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 + index * 0.1 }}
                  >
                    <div className={`absolute inset-0 bg-gradient-to-br ${item.gradient} opacity-0 group-hover:opacity-5 transition-opacity`} />
                    <div className="text-3xl mb-3">{item.icon}</div>
                    <div className="text-xs text-gray-500 mb-1 tracking-wider uppercase">{item.label}</div>
                    <div className="text-gray-900">{item.value}</div>
                  </motion.div>
                ))}
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty State */}
        {!generatedImage && !isGenerating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-center mt-16"
          >
            <motion.div
              className="inline-flex size-40 mx-auto mb-8 rounded-full bg-gradient-to-br from-amber-100 via-rose-100 to-purple-100 border-4 border-dashed border-purple-300/50 items-center justify-center shadow-inner"
              animate={{
                rotate: [0, 5, -5, 0],
              }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            >
              <Sparkles className="size-16 text-purple-400/60" />
            </motion.div>
          </motion.div>
        )}

        {/* Loading State */}
        {isGenerating && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center mt-16"
          >
            <div className="inline-flex size-32 mx-auto mb-6 rounded-full bg-gradient-to-br from-amber-400 via-rose-400 to-purple-500 items-center justify-center shadow-2xl">
              <Loader2 className="size-14 text-white animate-spin" />
            </div>
            <h3 className="text-2xl text-gray-700 mb-3">Creating Your Artwork...</h3>
            <p className="text-gray-500">
              {message}
            </p>
          </motion.div>
        )}
      </div>
    </section>
  );
}
