import type { AppConfig, Capabilities } from '../../types'

export function configsEqual(left: AppConfig, right: AppConfig): boolean {
  const clean = (value: AppConfig) => Object.fromEntries(
    Object.entries(value).filter(([key]) => !key.startsWith('_')),
  )
  return JSON.stringify(clean(left)) === JSON.stringify(clean(right))
}

export function capabilityLanguageCode(
  value: string,
  capabilities: Capabilities | null,
): string {
  const normalized = value.trim().toLocaleLowerCase()
  return capabilities?.languages.find(option => (
    option.value.toLocaleLowerCase() === normalized
    || option.label.replace(/\s*\([^)]*\)\s*$/, '').toLocaleLowerCase() === normalized
  ))?.value ?? normalized
}

export function sameModelFile(left: string, right: string): boolean {
  const fileName = (value: string) => (
    value.replaceAll('\\', '/').split('/').pop()?.toLocaleLowerCase() ?? ''
  )
  return Boolean(fileName(left) && fileName(left) === fileName(right))
}

export function contextWindowForSubtitles(value: number): number {
  const count = Math.min(5, Math.max(0, Math.trunc(value)))
  return 544 + (count * 160)
}

export function withCustomOption<T extends { label: string; value: string }>(
  options: T[],
  value: string,
) {
  if (options.some(option => option.value === value)) return options
  return [{ label: 'Custom', value, badge: value }, ...options]
}

export function modelFileName(value: string): string {
  return value.replaceAll('\\', '/').split('/').pop() || value
}
