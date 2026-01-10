import { motion } from 'motion/react';
import { Github, Mail, Linkedin, Palette, Heart, ExternalLink } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="relative bg-gradient-to-br from-slate-950 via-purple-950 to-rose-950 text-white overflow-hidden">
      {/* Artistic Background */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 20% 30%, rgba(251,191,36,0.2) 0%, transparent 50%),
                           radial-gradient(circle at 80% 70%, rgba(244,114,182,0.2) 0%, transparent 50%),
                           radial-gradient(circle at 50% 50%, rgba(168,85,247,0.15) 0%, transparent 50%)`,
        }} />
      </div>

      {/* Animated Gradient Orbs */}
      <motion.div
        className="absolute top-0 right-0 size-[40rem] bg-gradient-to-br from-amber-500/10 to-transparent rounded-full blur-3xl"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.15, 0.1],
        }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute bottom-0 left-0 size-[40rem] bg-gradient-to-tl from-purple-500/10 to-transparent rounded-full blur-3xl"
        animate={{
          scale: [1, 1.15, 1],
          opacity: [0.1, 0.2, 0.1],
        }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
      />

      {/* Top Decorative Border */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-amber-400 via-rose-400 to-purple-500" />

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-20">
        {/* Main Content Grid */}
        <div className="grid md:grid-cols-12 gap-12 mb-16">
          {/* Brand Section - Larger */}
          <motion.div 
            className="md:col-span-5"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="flex items-center gap-4 mb-6">
              <motion.div 
                className="size-16 bg-gradient-to-br from-amber-400 via-rose-400 to-purple-500 rounded-2xl flex items-center justify-center shadow-2xl"
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
              >
                <Palette className="size-8 text-white" />
              </motion.div>
              <div>
                <h3 className="text-2xl tracking-tight">StyleGAN</h3>
                <p className="text-purple-300 text-sm">Impressionist Art Generator</p>
              </div>
            </div>
            
            <p className="text-purple-200 leading-relaxed mb-6 text-lg">
              A project about combining the state-of-the-art of different disciplines and different centuries...
            </p>

            <div className="flex items-center gap-2 text-purple-300 text-sm">
              <Heart className="size-4 text-rose-400 fill-rose-400" />
              <span>Crafted with passion for art and technology</span>
            </div>
          </motion.div>

          {/* About Creator Section */}
          <motion.div 
            className="md:col-span-4"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <h3 className="text-xl mb-4 text-white">About the Creator</h3>
            <p className="text-purple-200 leading-relaxed mb-6">
              Alexander Grabner is a student at TU Wien with a great interest in Machine Learning, Computer Vision, and anything AI-related.
              He worked on the <span classname="text-amber-300">Impressionist StyleGAN</span> project from 10/2025 - 12/2025.
            </p>
            
            <div className="space-y-2">
              <a 
                href="#" 
                className="flex items-center gap-2 text-purple-300 hover:text-white transition-colors group"
              >
                <ExternalLink className="size-4 group-hover:translate-x-1 transition-transform" />
                <span className="text-sm">Read the technical report</span>
              </a>
              <a 
                href="#" 
                className="flex items-center gap-2 text-purple-300 hover:text-white transition-colors group"
              >
                <ExternalLink className="size-4 group-hover:translate-x-1 transition-transform" />
                <span className="text-sm">View source code</span>
              </a>
            </div>
          </motion.div>
        </div>

        {/* Divider with Gradient */}
        <div className="relative h-px mb-12">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-400/30 to-transparent" />
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-amber-400/50 via-rose-400/50 to-purple-400/50"
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1.5, ease: "easeInOut" }}
          />
        </div>

        {/* Bottom Section */}
        <div className="flex flex-col md:flex-row justify-between items-center gap-6 mb-8">
          <motion.div 
            className="flex items-center gap-3 text-purple-300 text-sm"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <motion.div 
              className="size-2 rounded-full bg-gradient-to-r from-green-400 to-emerald-500"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [1, 0.5, 1],
              }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <p>© {new Date().getFullYear()} StyleGAN Impressionist Generator by Alexander Grabner</p>
          </motion.div>
          
          
        </div>

        {/* Inspirational Quote */}
        <motion.div 
          className="text-center"
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.5 }}
        >
          <div className="inline-block px-8 py-4 bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10">
            <p className="text-purple-200/80 text-sm italic leading-relaxed max-w-2xl">
              "Color is my day-long obsession, joy and torment."
            </p>
            <p className="mt-2 text-xs text-purple-300">— Claude Monet</p>
          </div>
        </motion.div>

        {/* Decorative Bottom Elements */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-3/4 h-px bg-gradient-to-r from-transparent via-purple-500/20 to-transparent" />
      </div>

      {/* Bottom Gradient Bar */}
      <motion.div 
        className="h-2 bg-gradient-to-r from-amber-500 via-rose-500 to-purple-600"
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      />
    </footer>
  );
}
