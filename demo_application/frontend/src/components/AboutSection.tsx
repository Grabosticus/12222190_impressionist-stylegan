import { motion } from 'motion/react';
import { Palette, Eye, Lightbulb, Waves, Brush, Sun, Wind } from 'lucide-react';

export default function AboutSection() {
  return (
    <section className="relative bg-gradient-to-b from-purple-50 via-white to-amber-50 py-32 px-4 overflow-hidden">
      {/* Decorative Top Border */}
      <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-amber-400 via-rose-400 to-purple-500" />
      
      {/* Floating Background Elements */}
      <div className="absolute inset-0 opacity-20">
        <motion.div
          className="absolute top-40 right-20 size-96 bg-gradient-to-br from-amber-200 to-orange-200 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            x: [0, 30, 0],
            y: [0, -30, 0],
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute bottom-40 left-20 size-96 bg-gradient-to-tl from-purple-200 to-blue-200 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.15, 1],
            x: [0, -30, 0],
            y: [0, 30, 0],
          }}
          transition={{ duration: 25, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>
      
      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <motion.div 
          className="text-center mb-24"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <motion.div 
            className="inline-flex items-center gap-2 mb-6 px-6 py-3 rounded-full bg-white/80 backdrop-blur-xl border-2 border-purple-200 shadow-lg"
            whileHover={{ scale: 1.05 }}
          >
            <Brush className="size-4 text-purple-600" />
            <span className="text-sm tracking-widest text-purple-900">THE INTERSECTION OF ART & AI</span>
          </motion.div>
          
          <h2 className="mb-6 text-5xl md:text-6xl lg:text-7xl bg-gradient-to-r from-amber-600 via-rose-600 to-purple-700 bg-clip-text text-transparent leading-tight">
            About Impressionism
          </h2>
          
          <p className="text-xl text-gray-700 max-w-3xl mx-auto leading-relaxed font-light">
            Discover the art of Impressionism and its defining characteristics.
          </p>
        </motion.div>

        {/* Feature Cards Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-24">
          {[
            {
              icon: Palette,
              title: 'Color',
              description: 'Impressionism was always about color. Artists tried to capture the feelings and essence of different shades of blues, reds, and others to create art which speaks not through what it depicts, but how it does so.',
              gradient: 'from-amber-400 to-orange-500',
              bgGradient: 'from-amber-50 to-orange-50',
              delay: 0.1
            },
            {
              icon: Sun,
              title: 'Light',
              description: 'A defining characteristic of Impressionist art is its ability to capture the effects of light. Colors have often been chosen to reflect the natural light of the scenery, and to capture fleeting light effects by placing complementary colors side by side.',
              gradient: 'from-rose-400 to-pink-500',
              bgGradient: 'from-rose-50 to-pink-50',
              delay: 0.2
            },
            {
              icon: Waves,
              title: 'Brushwork',
              description: 'Impressionisn was revolutionary in its artistic style. An important reason for this was its brushwork. Impressionist painters strayed away from trying to create sharp, chiseled faces or realistic looking trees. Rather they used loose brushwork to create an almost dream-like scenery.',
              gradient: 'from-purple-400 to-indigo-500',
              bgGradient: 'from-purple-50 to-indigo-50',
              delay: 0.3
            },
            {
              icon: Wind,
              title: 'Movement',
              description: 'Impressionist painters often tried to capture the movement of objects in their scenery. A beautiful example of this is Claude Monets "Madame Monet and Her Son", which shows Madame Monet turning on a windy field the moment the scenery was captured.',
              gradient: 'from-blue-400 to-cyan-500',
              bgGradient: 'from-blue-50 to-cyan-50',
              delay: 0.4
            },
          ].map((feature, index) => (
            <motion.div
              key={feature.title}
              className={`relative bg-gradient-to-br ${feature.bgGradient} to-white p-8 rounded-3xl border-2 border-white shadow-xl hover:shadow-2xl transition-all group overflow-hidden`}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.3, delay: feature.delay }}
              whileHover={{ y: -8 }}
            >
              {/* Hover Gradient Overlay */}
              <motion.div
                className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-5 transition-opacity duration-500`}
              />
              
              {/* Icon */}
              <motion.div 
                className={`inline-flex size-16 rounded-2xl bg-gradient-to-br ${feature.gradient} items-center justify-center mb-6 shadow-lg`}
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.6 }}
              >
                <feature.icon className="size-8 text-white" />
              </motion.div>
              
              <h3 className="mb-4 text-xl text-gray-900">{feature.title}</h3>
              <p className="text-gray-700 text-sm leading-relaxed">
                {feature.description}
              </p>
              
              {/* Decorative Element */}
              <div className={`absolute bottom-4 right-4 size-20 bg-gradient-to-br ${feature.gradient} opacity-10 rounded-full blur-xl`} />
            </motion.div>
          ))}
        </div>

        {/* Main Content Card */}
        <motion.div
          className="relative bg-gradient-to-br from-purple-900 via-rose-900 to-amber-900 text-white rounded-[2.5rem] shadow-2xl overflow-hidden"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          {/* Texture Overlay */}
          <div className="absolute inset-0 opacity-10" style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h60v60H0z' fill='none'/%3E%3Cpath d='M30 30m-20 0a20 20 0 1 0 40 0a20 20 0 1 0-40 0' fill='%23fff' fill-opacity='0.05'/%3E%3C/svg%3E")`,
          }} />
          
          {/* Gradient Orbs */}
          <div className="absolute top-0 right-0 size-96 bg-gradient-to-br from-amber-500/20 to-transparent rounded-full blur-3xl" />
          <div className="absolute bottom-0 left-0 size-96 bg-gradient-to-tl from-purple-500/20 to-transparent rounded-full blur-3xl" />
          
          <div className="relative z-10 p-10 md:p-16">
            {/* Header with Icon */}
            <div className="flex items-start gap-6 mb-10">
              <motion.div 
                className="flex-shrink-0 size-20 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center border border-white/30"
                whileHover={{ rotate: 180 }}
                transition={{ duration: 0.6 }}
              >
                <Eye className="size-10 text-white" />
              </motion.div>
              <div>
                <h3 className="text-3xl md:text-4xl mb-3 text-white">Impressionist <span className="text-rose-300">StyleGAN</span></h3>
                <p className="text-purple-200">A Project by <span className="text-amber-300">Alexander Grabner</span></p>
              </div>
            </div>
            
            {/* Content */}
            <div className="space-y-6 mb-10">
              <p className="text-lg text-purple-100 leading-relaxed">
                I thought long about what type of Machine Learning project I would create for the course but ultimately decided
                on an Image Generator for artworks. I have always been fascinated by these models, which seem to so seamlessly be able to
                accomplish such a highly creative task as creating an image. I knew however, that not every artstyle is suitable for 
                a "low-budget" StyleGAN. I searched for an artstyle that has many similar looking, not to precise, but still beautiful and
                intriguing artworks. Impressionism fulfills all of these requirements. Painters such as <span classname="text-rose-300">Claude Monet </span> 
                with his <span class="text-amber-300">Water Lillies</span> are known for having produced many different versions of the same painting.
                Also, Impessionism, with its dreamy atmosphere and imprecise brushwork, seemed like it would be way easier learnable by a machine than other
                more "precise" styles.
              </p>
              
              <p className="text-lg text-purple-100 leading-relaxed">
                I collected 23,328 impressionist, post-impressionist, and neo-impressionist artworks from public sources and used them to create one of the 
                most extensive datasets of curated impressionist art today. You can find the dataset in the <span classname="text-rose-300">Impressionist Artworks v1.0 </span> 
                GitHub Release (find the link in the footer section). Each dataset entry contains an URL to the artwork image as well as metadata, 
                such as the genre (e.g. still life, landscape) or the name of the artwork, which you could use to train all kinds of models on the dataset.
              </p>
              
              <p className="text-lg text-purple-100 leading-relaxed">
                The StyleGAN I trained, is in its essence a first-generation StyleGAN. I did however incorporate techniques found in later StyleGAN architectures to improve 
                the quality of the generated images. For example, my StyleGAN features an <span classname="text-rose-300">Adaptive Discriminator Augmentation</span> mechanism, 
                similar to <span classname="text-amber-300">StyleGAN-2 ADA</span>, but still contains an <span classname="text-amber-300">AdaIN</span> module and <span classname="text-amber-300">Progressive Growing </span>
                found in the original architecture. You can find the code of my model in the GitHub repository as well as some pretrained weights in the <span classname="text-rose-300">Impressionist Artworks v1.0</span> release.
                My final model achieves an FID score of 40.38 on a subset of landscape paintings of the dataset. FID as a metric isn't perfect - its values depend on the size of the dataset. 
                I think if my dataset was larger, the FID score would have been lower, even with the same quality of generated images.
              </p>
            </div>

            {/* Stats Grid */}
            <div className="grid sm:grid-cols-3 gap-6">
              {[
                { emoji: '🎨', label: 'Training Artworks', value: '23,328', subtext: 'Diverse collection of paintings' },
                { emoji: '🧑‍🎨', label: 'Artists', value: '100+', subtext: 'Many different artists with different styles' },
                { emoji: '📊', label: 'Image Quality', value: 'FID 40.38', subtext: 'Achieved on a subset of similar images' },
              ].map((stat, index) => (
                <motion.div
                  key={stat.label}
                  className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20"
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
                  whileHover={{ scale: 1.05, backgroundColor: 'rgba(255,255,255,0.15)' }}
                >
                  <div className="text-4xl mb-3">{stat.emoji}</div>
                  <div className="text-xs text-purple-300 mb-1 uppercase tracking-wider">{stat.label}</div>
                  <div className="text-2xl mb-1 text-white">{stat.value}</div>
                  <div className="text-xs text-purple-200">{stat.subtext}</div>
                </motion.div>
              ))}
            </div>

            {/* Famous Artists Section */}
            <motion.div 
              className="mt-12 p-6 bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <h4 className="text-lg mb-4 text-purple-200 flex items-center gap-2">
                <Brush className="size-5" />
                Trained on Works By
              </h4>
              <div className="flex flex-wrap gap-3">
                {[
                  'Claude Monet', 'Pierre-Auguste Renoir', 'Camille Pissarro', 
                  'Edgar Degas', 'Alfred Sisley', 'Berthe Morisot', 'Mary Cassatt', 'Many Many More...'
                ].map((artist, index) => (
                  <motion.span
                    key={artist}
                    className="px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full text-sm text-white border border-white/20"
                    initial={{ opacity: 0, scale: 0.8 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.4, delay: 0.9 + index * 0.05 }}
                    whileHover={{ scale: 1.1, backgroundColor: 'rgba(255,255,255,0.2)' }}
                  >
                    {artist}
                  </motion.span>
                ))}
              </div>
            </motion.div>
          </div>
        </motion.div>

        {/* Bottom Quote */}
        <motion.div
          className="mt-16 text-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <blockquote className="text-2xl md:text-3xl text-gray-600 italic max-w-4xl mx-auto leading-relaxed">
            "I perhaps owe having become a painter to flowers."
          </blockquote>
          <p className="mt-4 text-lg text-gray-500">— Claude Monet</p>
        </motion.div>
      </div>
    </section>
  );
}
