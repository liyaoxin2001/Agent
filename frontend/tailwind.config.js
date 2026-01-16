/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 暗色模式颜色
        'chat-bg': '#343541',
        'chat-bg-light': '#444654',
        'sidebar-bg': '#202123',
        'user-msg': '#343541',
        'assistant-msg': '#444654',
        // 亮色模式颜色
        'light-chat-bg': '#ffffff',
        'light-chat-bg-light': '#f7f7f8',
        'light-sidebar-bg': '#f7f7f8',
        'light-user-msg': '#ffffff',
        'light-assistant-msg': '#f7f7f8',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-in',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
  darkMode: 'class',
}
