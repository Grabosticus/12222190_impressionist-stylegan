import { motion } from 'motion/react';
import { ArrowDown, Sparkles } from 'lucide-react';

export default function LandingSection() {
  return (
    <section className="relative h-screen flex items-center justify-center overflow-hidden">
      {/* Background Image with Parallax Effect */}
      <motion.div 
        className="absolute inset-0 bg-cover bg-center scale-110"
        style={{
          backgroundImage: `url('https://images.unsplash.com/photo-1763491905762-66f2d0bf57ac?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb25ldCUyMHdhdGVyJTIwbGlsaWVzJTIwcGFpbnRpbmd8ZW58MXx8fHwxNzY1NDAzNTI0fDA&ixlib=rb-4.1.0&q=80&w=1080')`,
        }}
        initial={{ scale: 1.1 }}
        animate={{ scale: 1.05 }}
        transition={{ duration: 20, repeat: Infinity, repeatType: "reverse" }}
      />
      
      {/* Multi-layer Gradient Overlay for Depth */}
      <div className="absolute inset-0 bg-gradient-to-b from-indigo-950/85 via-purple-900/75 to-rose-950/85" />
      <div className="absolute inset-0 bg-gradient-to-tr from-amber-900/40 via-transparent to-blue-900/40" />
      
      {/* Artistic Texture Overlay */}
      <div className="absolute inset-0 opacity-20 mix-blend-overlay" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' /%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.4'/%3E%3C/svg%3E")`,
      }} />
      
      {/* Painterly Light Effects */}
      <div className="absolute inset-0 opacity-30">
        <motion.div 
          className="absolute top-20 left-20 size-96 bg-gradient-to-br from-amber-300 to-orange-400 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div 
          className="absolute bottom-20 right-32 size-[32rem] bg-gradient-to-tl from-blue-400 to-purple-500 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.3, 0.4, 0.3],
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />
        <motion.div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 size-[28rem] bg-gradient-to-br from-rose-400 to-pink-500 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.15, 1],
            opacity: [0.2, 0.35, 0.2],
          }}
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut", delay: 2 }}
        />
      </div>
      
      {/* Content */}
      <div className="relative z-10 text-center text-white px-4 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.3, ease: "easeOut" }}
        >
          <motion.div 
            className="inline-flex items-center gap-2 mb-10 px-6 py-3 rounded-full bg-white/10 backdrop-blur-md border border-white/30 shadow-2xl"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <Sparkles className="size-4 text-amber-300" />
            <span className="text-sm tracking-widest bg-gradient-to-r from-amber-200 via-rose-200 to-purple-200 bg-clip-text text-transparent">
              AI GENERATED IMPRESSIONIST ARTWORKS
            </span>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
          className="mb-10"
        >
          <h1 className="text-7xl md:text-8xl lg:text-9xl leading-[0.95] tracking-tight mb-6">
            <motion.span 
              className="block bg-gradient-to-r from-amber-200 via-rose-200 to-orange-200 bg-clip-text text-transparent drop-shadow-2xl"
              animate={{
                backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
              }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              style={{ backgroundSize: '200% 200%' }}
            >
              StyleGAN
            </motion.span>
            <motion.span 
              className="block mt-3 bg-gradient-to-r from-white via-purple-100 to-blue-100 bg-clip-text text-transparent"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 1, delay: 0.8 }}
            >
              Impressionist
            </motion.span>
          </h1>
          
          {/* Decorative Line with Dots */}
          <div className="flex items-center justify-center gap-3 mt-8">
            <motion.div 
              className="h-px w-16 bg-gradient-to-r from-transparent to-amber-300"
              initial={{ width: 0 }}
              animate={{ width: 64 }}
              transition={{ duration: 1, delay: 1 }}
            />
            <motion.div 
              className="size-2 rounded-full bg-amber-300"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: 1.2 }}
            />
            <motion.div 
              className="size-2 rounded-full bg-rose-300"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: 1.3 }}
            />
            <motion.div 
              className="size-2 rounded-full bg-purple-300"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: 1.4 }}
            />
            <motion.div 
              className="h-px w-16 bg-gradient-to-l from-transparent to-purple-300"
              initial={{ width: 0 }}
              animate={{ width: 64 }}
              transition={{ duration: 1, delay: 1 }}
            />
          </div>
        </motion.div>

        <motion.p 
          className="mb-14 text-xl md:text-2xl text-purple-100/90 max-w-3xl mx-auto leading-relaxed font-light"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.9 }}
        >
          Imprecise brushwork meets probability distributions, details meet noise injection, canvases meet progressive growing...
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1.1 }}
        >
          <motion.button 
            onClick={() => {
              document.getElementById('generate-section')?.scrollIntoView({ 
                behavior: 'smooth' 
              });
            }}
            className="group relative bg-gradient-to-r from-amber-400 via-rose-400 to-purple-500 text-white px-12 py-5 rounded-full transition-all shadow-2xl shadow-purple-500/30 overflow-hidden"
            whileHover={{ scale: 1.05, boxShadow: "0 25px 50px -12px rgba(168, 85, 247, 0.5)" }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="relative z-10 flex items-center gap-3 tracking-wide">
              Generate Your Own Images
              <motion.div
                animate={{ y: [0, 3, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <ArrowDown className="size-5" />
              </motion.div>
            </span>
            <motion.div 
              className="absolute inset-0 bg-gradient-to-r from-amber-300 via-rose-300 to-purple-400"
              initial={{ opacity: 0 }}
              whileHover={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            />
          </motion.button>
        </motion.div>

        {/* Floating Artistic Elements */}
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute size-3 rounded-full bg-white/40 backdrop-blur-sm"
            style={{
              left: `${15 + i * 15}%`,
              top: `${20 + (i % 3) * 25}%`,
            }}
            animate={{
              y: [0, -30, 0],
              opacity: [0.4, 0.8, 0.4],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 4 + i * 0.5,
              repeat: Infinity,
              delay: i * 0.3,
              ease: "easeInOut"
            }}
          />
        ))}
      </div>

      {/* Elegant Scroll Indicator */}
      <motion.div 
        className="absolute bottom-12 left-1/2 -translate-x-1/2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
      >
        <motion.div 
          animate={{ y: [0, 12, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="flex flex-col items-center gap-3"
        >
          <div className="text-xs tracking-[0.3em] text-purple-200/60 uppercase">Scroll</div>
          <div className="size-8 rounded-full border-2 border-purple-200/40 flex items-center justify-center backdrop-blur-sm">
            <ArrowDown className="size-4 text-purple-200/60" />
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}
