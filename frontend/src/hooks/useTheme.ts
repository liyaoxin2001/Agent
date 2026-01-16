import { useState, useEffect } from 'react'

type Theme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'huahuaChat_theme'

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    // 从 localStorage 读取主题，如果没有则默认为 dark
    const stored = localStorage.getItem(THEME_STORAGE_KEY) as Theme | null
    return stored || 'dark'
  })

  useEffect(() => {
    const root = document.documentElement
    
    if (theme === 'dark') {
      root.classList.add('dark')
      root.classList.remove('light')
    } else {
      root.classList.remove('dark')
      root.classList.add('light')
    }
    
    // 保存到 localStorage
    localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  return { theme, toggleTheme, setTheme }
}
