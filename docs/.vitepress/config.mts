import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Stalzy's Blog",
  description: '个人博客 - 分享技术与思考',
  lang: 'zh-CN',
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '博客', link: '/posts/' }
    ],

    sidebar: {
      '/posts/': [
        {
          text: '博客文章',
          items: [
            { text: 'HealthBench：从框架理解到实验探索', link: '/posts/healthbench' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/stalzy' }
    ],

    footer: {
      message: '基于 VitePress 构建',
      copyright: `© ${new Date().getFullYear()} Stalzy`
    },

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
      label: '目录'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    }
  }
})
